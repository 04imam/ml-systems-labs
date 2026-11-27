from run_lab7 import *

def resume():
    cfg = yaml.safe_load(
        (ROOT / "config.yaml").read_text(encoding="utf-8")
    )

    seed_everything(cfg["seed"])

    dev = "cuda" if torch.cuda.is_available() and cfg["allow_cuda"] else "cpu"

    train_loader, val_loader, test_loader = loaders(cfg)

    rows = json.loads(
        (ROOT / "results" / "metrics.json").read_text()
    )

    completed = {r["model"] for r in rows}

    history_path = ROOT / "results" / "training_histories.json"

    log = json.loads(history_path.read_text()) if history_path.exists() else {}

    teacher = CNN(
        cfg["teacher_channels"],
        cfg["teacher_hidden"]
    )

    teacher.load_state_dict(
        torch.load(
            ROOT / "checkpoints" / "teacher.pt",
            map_location="cpu",
            weights_only=True
        )
    )

    teacher.eval()

    def save_result(model, name, group, sparsity=0.0):
        rows.append(record(
            model.cpu(), name, group,
            test_loader, cfg, sparsity
        ))
        write_rows(rows)
        history_path.write_text(
            json.dumps(log, indent=2),
            encoding="utf-8"
        )
        completed.add(name)
        print("SAVED:", name, flush=True)

    print("Teacher checkpoint loaded successfully.")
    print("Completed experiments:", len(completed))

    for mode in ("global", "layerwise"):
        for sparsity in cfg["sparsities"]:
            name = f'{mode}_{str(sparsity).replace(".", "p")}'

            if name in completed:
                print("SKIPPING:", name, flush=True)
                continue

            pruned = copy.deepcopy(teacher)
            masks = unstructured_mask(pruned, sparsity, mode)
            apply_masks(pruned, masks)
            pruned = pruned.to(dev)

            log[name] = train(
                pruned, train_loader, val_loader,
                dev, cfg["finetune_epochs"],
                cfg["finetune_lr"],
                masks={k: v.to(dev) for k, v in masks.items()},
                tag=name
            )

            save_result(pruned, name, mode, sparsity)

    for level in cfg["structured_levels"]:
        name = f'structured_{str(level).replace(".", "p")}'

        if name in completed:
            print("SKIPPING:", name, flush=True)
            continue

        pruned = structured_channel_prune(
            teacher, level
        ).to(dev)

        log[name] = train(
            pruned, train_loader, val_loader,
            dev, cfg["finetune_epochs"],
            cfg["finetune_lr"], tag=name
        )

        save_result(pruned, name, "structured", level)

    for distill in (False, True):
        name = "student_distilled" if distill else "student_plain"

        if name in completed:
            print("SKIPPING:", name, flush=True)
            continue

        seed_everything(cfg["seed"] + 101)

        student = CNN(
            cfg["student_channels"],
            cfg["student_hidden"]
        ).to(dev)

        t = copy.deepcopy(teacher).to(dev) if distill else None

        log[name] = train(
            student, train_loader, val_loader,
            dev, cfg["student_epochs"], cfg["lr"],
            teacher=t, alpha=cfg["alpha"],
            T=cfg["temperature"], tag=name
        )

        save_result(student, name, "student")

    plots(rows)
    make_report(rows, cfg, dev)

    print("LAB 7 COMPLETED", flush=True)

if __name__ == "__main__":
    resume()
