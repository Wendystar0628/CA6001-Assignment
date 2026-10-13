"""One CLI for preparation, training, honest replay and output export."""
import argparse
import json
from pathlib import Path
from .config import load_config

def main(argv=None):
    parser = argparse.ArgumentParser(description="CA6001 CPT budget experiments")
    parser.add_argument("command", choices=["restore-inputs", "prepare", "split", "pilot", "fit", "freeze-quality",
                                           "evaluate-static", "replay", "summarize", "export", "run", "status"])
    parser.add_argument("--config")
    parser.add_argument("--run", dest="run_id")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--site")
    parser.add_argument("--models", help="Comma-separated model IDs; explicit subsets are recorded")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preset", choices=["both", "report", "slides"], default="both")
    parser.add_argument("--figure")
    args = parser.parse_args(argv)
    config = load_config(args.config or ("configs/pilot.yaml" if args.command == "pilot" else "configs/main.yaml"))
    # Output-only stages reopen the saved experiment configuration, including
    # its explicit model subset. No training rerun is needed to redraw a figure.
    if args.config is None and args.command in ("summarize", "export", "status"):
        output = Path(config["_root"]) / config["project"]["output_dir"]
        if args.run_id is None:
            args.run_id = json.loads((output / "latest.json").read_text())["run_id"]
        stored = json.loads((output / args.run_id / "run.json").read_text(encoding="utf-8"))
        config = {**stored["config"], "_root": config["_root"]}
    if args.command == "pilot":
        config.setdefault("run", {})["mode"] = "pilot"
        config.setdefault("tabpfn", {})["max_steps"] = 1
        config["tabpfn"]["max_epochs"] = 1
    models = args.models.split(",") if args.models else None
    if models:
        from .pipeline import MODELS
        if set(models) - set(MODELS):
            parser.error("Unknown model ID")
        config.setdefault("run", {})["models"] = models
    from . import pipeline
    if args.command == "restore-inputs":
        from .data.import_existing import restore_arrays
        result = restore_arrays(config)
    elif args.command == "prepare":
        result = pipeline.prepare(config)
    elif args.command == "split":
        result = pipeline.split(config)
    elif args.command == "status":
        output = Path(config["_root"]) / config["project"]["output_dir"]
        run_id = args.run_id or json.loads((output / "latest.json").read_text())["run_id"]
        result = json.loads((output / run_id / "run.json").read_text(encoding="utf-8"))
    else:
        directory = pipeline.open_run(config, args.run_id, resume=args.resume or args.command not in ("pilot", "run"))
        print(f"Run: {directory}", flush=True)
        if args.command == "pilot":
            result = pipeline.pilot(config, directory, site_id=args.site, models=models)
        elif args.command == "fit":
            result = pipeline.fit(config, directory, args.fold, models=models)
        elif args.command == "freeze-quality":
            result = pipeline.freeze_quality(config, directory, args.fold)
        elif args.command in ("evaluate-static", "replay"):
            policies = ["maximin"] if args.command == "evaluate-static" else None
            result = pipeline.replay(config, directory, fold_id=args.fold, models=models, policies=policies, site_id=args.site)
        elif args.command == "summarize":
            result = pipeline.summarize(config, directory)
        elif args.command == "export":
            result = pipeline.export(config, directory, args.preset, args.figure)
        else:
            pipeline._ensure_prepared(config)
            pipeline.fit(config, directory, args.fold, models=models)
            pipeline.freeze_quality(config, directory, args.fold)
            pipeline.replay(config, directory, fold_id=args.fold, models=models)
            pipeline.summarize(config, directory)
            result = pipeline.export(config, directory, args.preset)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
