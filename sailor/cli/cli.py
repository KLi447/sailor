import argparse
import requests
from pathlib import Path
import sys


def submit_job(server_url: str, r: int, alpha: int, model_name: str,
               target_modules: list[str], dataset_path: Path):

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    url = f"{server_url.rstrip('/')}/submit_job"

    files = {"dataset": open(dataset_path, "rb")}
    data = {
        "r": r,
        "alpha": alpha,
        "model_name": model_name,
        "target_modules": ",".join(target_modules)
    }

    print(f"Sending request to {url} ...")
    try:
        response = requests.post(url, data=data, files=files, timeout=60)
        response.raise_for_status()
        print("Job submitted successfully!\n")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Failed to submit job: {e}")


def interactive_mode(default_url="http://127.0.0.1:8000"):
    print("\n🔧 LoRA Fine-tuning Task Submission\n")

    server_url = input(f"Server URL [{default_url}]: ").strip() or default_url
    r = int(input("LoRA rank (r): ").strip())
    alpha = int(input("Scaling factor (alpha): ").strip())
    model_name = input("Base model name (e.g., llama-7b): ").strip()
    target_modules = input("Target modules (comma-separated): ").strip().split(",")
    dataset_path = Path(input("Path to dataset JSON: ").strip())

    submit_job(server_url, r, alpha, model_name, target_modules, dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="CLI client to submit LoRA fine-tuning tasks to the server."
    )
    parser.add_argument("--server", "-s", type=str,
                        default="http://127.0.0.1:8000",
                        help="Base URL of the LoRA server")
    parser.add_argument("--r", type=int, help="LoRA rank")
    parser.add_argument("--alpha", type=int, help="Scaling factor")
    parser.add_argument("--model", type=str, help="Base model name (e.g. llama-7b)")
    parser.add_argument("--target-modules", type=str,
                        help="Comma-separated target modules, e.g. q_proj,v_proj")
    parser.add_argument("--dataset", type=Path, help="Path to dataset JSON file")

    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive prompt mode")

    args = parser.parse_args()

    if args.interactive or not all([args.r, args.alpha, args.model, args.target_modules, args.dataset]):
        interactive_mode(args.server)
    else:
        submit_job(
            server_url=args.server,
            r=args.r,
            alpha=args.alpha,
            model_name=args.model,
            target_modules=args.target_modules.split(","),
            dataset_path=args.dataset
        )


if __name__ == "__main__":
    main()
