import torch

def check_gpu():
    if torch.cuda.is_available():
        print("[PASS] ROCm-kompatible GPU erkannt.")
        print(f"Gefundene GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f" - Gerät {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[FAIL] Keine GPU erkannt oder ROCm nicht aktiv.")

if __name__ == "__main__":
    check_gpu()