import argparse
import os
import subprocess
import threading
from queue import Empty, Queue
from pathlib import Path


def run(cmd, env=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_subjects(subject_id, csv_path):
    if subject_id:
        return [subject_id]
    if not csv_path:
        raise ValueError("Provide either --subject-id or --csv.")
    subjects = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    for row in rows[1:]:
        parts = row.strip().split(",")
        if parts and parts[0]:
            subjects.append(parts[0])
    return subjects


def parse_list(text):
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def detect_gpu_ids(requested_gpu_ids):
    if requested_gpu_ids:
        return parse_list(requested_gpu_ids)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return parse_list(visible)

    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        gpu_ids = [line.strip() for line in result.splitlines() if line.strip()]
        if gpu_ids:
            return gpu_ids
    except Exception:
        pass

    return ["0"]


def build_subject_cmd(args, subject):
    cmd = [
        "python",
        "-m",
        "atlasgs.experiments.run_subject",
        "--subject-id",
        subject,
        "--data-root",
        str(Path(args.data_root)),
        "--out-root",
        str(Path(args.out_root)),
        "--medgs-root",
        str(Path(args.medgs_root)),
        "--target-modalities",
        str(args.target_modalities),
        "--factor-z",
        str(args.factor_z),
        "--interp-factor",
        str(args.interp_factor),
        "--train-iterations",
        str(args.train_iterations),
        "--t1guided-iterations",
        str(args.t1guided_iterations),
        "--t1guided-lambda-dssim",
        str(args.t1guided_lambda_dssim),
        "--t1guided-time-lr",
        str(args.t1guided_time_lr),
        "--alpine-a2-pretrain-iters",
        str(args.alpine_a2_pretrain_iters),
        "--alpine-a2-fit-iters",
        str(args.alpine_a2_fit_iters),
        "--alpine-a2-batch-size",
        str(args.alpine_a2_batch_size),
        "--alpine-a2-chunk-size",
        str(args.alpine_a2_chunk_size),
        "--seed",
        str(args.seed),
    ]
    if args.apply_target_brain_mask:
        cmd.append("--apply-target-brain-mask")
    if args.t1guided_time_init_auto:
        cmd.append("--t1guided-time-init-auto")
    if args.run_t1guided_latent:
        cmd.extend(
            [
                "--run-t1guided-latent",
                "--t1guided-latent-iterations",
                str(args.t1guided_latent_iterations),
            ]
        )
    if args.run_t1guided_topology:
        cmd.extend(
            [
                "--run-t1guided-topology",
                "--t1guided-topology-iterations",
                str(args.t1guided_topology_iterations),
                "--t1guided-topology-lambda-dssim",
                str(args.t1guided_topology_lambda_dssim),
                "--t1guided-topology-interp-lambda-dssim",
                str(args.t1guided_topology_interp_lambda_dssim),
                "--t1guided-topology-persist-lambda",
                str(args.t1guided_topology_persist_lambda),
            ]
        )
    if args.run_t1guided_our:
        cmd.extend(
            [
                "--run-t1guided-our",
                "--t1guided-our-iterations",
                str(args.t1guided_our_iterations),
                "--t1guided-our-lambda-dssim",
                str(args.t1guided_our_lambda_dssim),
            ]
        )
    if args.t1guided_refine_t1:
        cmd.extend(
            [
                "--t1guided-refine-t1",
                "--t1guided-refine-t1-iterations",
                str(args.t1guided_refine_t1_iterations),
            ]
        )
    if args.run_lr_fusion:
        cmd.append("--run-lr-fusion")
    if args.no_run_alpine_a2:
        cmd.append("--no-run-alpine-a2")
    if args.no_run_t1guided_our_fusion:
        cmd.append("--no-run-t1guided-our-fusion")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_subject(args, subject, gpu_id):
    cmd = build_subject_cmd(args, subject)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[SUBJECT] {subject} gpu={gpu_id}")
    run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Run full method suite: interp, ALPINE INR, single MedGS, clean T1-guided, latent, topology, LR-fusion, and our combined method."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--medgs-root", required=True)
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--target-modalities", type=str, default="flair")
    parser.add_argument("--factor-z", type=int, default=7)
    parser.add_argument("--interp-factor", type=int, default=7)
    parser.add_argument("--train-iterations", type=int, default=30000)
    parser.add_argument("--t1guided-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-latent-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-topology-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-our-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-refine-t1-iterations", type=int, default=10000)
    parser.add_argument("--t1guided-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--t1guided-time-lr", type=float, default=5e-4)
    parser.add_argument("--t1guided-time-init-auto", action="store_true")
    parser.add_argument("--t1guided-topology-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--t1guided-topology-interp-lambda-dssim", type=float, default=-1.0)
    parser.add_argument("--t1guided-topology-persist-lambda", type=float, default=0.1)
    parser.add_argument("--t1guided-our-lambda-dssim", type=float, default=0.2)
    parser.add_argument("--alpine-a2-pretrain-iters", type=int, default=2000)
    parser.add_argument("--alpine-a2-fit-iters", type=int, default=4000)
    parser.add_argument("--alpine-a2-batch-size", type=int, default=32768)
    parser.add_argument("--alpine-a2-chunk-size", type=int, default=262144)
    parser.add_argument("--parallel-subjects", type=int, default=0, help="Concurrent subjects. <=0 means auto from GPU count.")
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids for subject-level parallel runs.")
    parser.add_argument("--apply-target-brain-mask", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-t1guided-latent", dest="run_t1guided_latent", action="store_true")
    parser.add_argument("--no-run-t1guided-latent", dest="run_t1guided_latent", action="store_false")
    parser.set_defaults(run_t1guided_latent=True)
    parser.add_argument("--run-t1guided-topology", dest="run_t1guided_topology", action="store_true")
    parser.add_argument("--no-run-t1guided-topology", dest="run_t1guided_topology", action="store_false")
    parser.set_defaults(run_t1guided_topology=True)
    parser.add_argument("--run-t1guided-our", dest="run_t1guided_our", action="store_true")
    parser.add_argument("--no-run-t1guided-our", dest="run_t1guided_our", action="store_false")
    parser.set_defaults(run_t1guided_our=True)
    parser.add_argument("--t1guided-refine-t1", action="store_true")
    parser.add_argument("--run-lr-fusion", dest="run_lr_fusion", action="store_true")
    parser.add_argument("--no-run-lr-fusion", dest="run_lr_fusion", action="store_false")
    parser.set_defaults(run_lr_fusion=True)
    parser.add_argument("--no-run-alpine-a2", action="store_true")
    parser.add_argument("--no-run-t1guided-our-fusion", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = parse_subjects(args.subject_id, args.csv)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    if len(subjects) == 0:
        raise ValueError("No subjects resolved from input.")

    gpu_ids = detect_gpu_ids(args.gpu_ids)
    requested_parallel = args.parallel_subjects if args.parallel_subjects > 0 else len(gpu_ids)
    workers = max(1, min(requested_parallel, len(gpu_ids), len(subjects)))
    worker_gpus = gpu_ids[:workers]
    print(
        f"[PARALLEL] subjects={len(subjects)} workers={workers} "
        f"gpu_ids={worker_gpus}"
    )

    if workers == 1:
        for subject in subjects:
            run_subject(args, subject, worker_gpus[0])
        return

    queue = Queue()
    for subject in subjects:
        queue.put(subject)

    errors = []
    errors_lock = threading.Lock()
    stop_flag = threading.Event()

    def worker_fn(gpu_id):
        while not stop_flag.is_set():
            try:
                subject = queue.get_nowait()
            except Empty:
                return
            try:
                run_subject(args, subject, gpu_id)
            except Exception as exc:
                with errors_lock:
                    errors.append((subject, gpu_id, exc))
                stop_flag.set()
                return
            finally:
                queue.task_done()

    threads = []
    for gpu_id in worker_gpus:
        thread = threading.Thread(target=worker_fn, args=(gpu_id,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    if errors:
        subject, gpu_id, exc = errors[0]
        raise RuntimeError(
            f"Subject {subject} failed on gpu {gpu_id}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
