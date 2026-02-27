import subprocess
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SystemState:
    kernel_version: str = ""
    suid_binaries: List[str] = None
    sudo_privileges: str = ""
    effective_user: str = "" # Added to prevent label leakage
    euid: int = 0
    
    def __post_init__(self):
        self.suid_binaries = self.suid_binaries or []

class LinuxReconEngine:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.state = SystemState()
        # Capture context immediately
        self.state.effective_user = os.getlogin() if hasattr(os, 'getlogin') else os.getenv('USER')
        self.state.euid = os.geteuid()

    def _execute_safe(self, command: List[str]) -> str:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            return result.stdout.strip()
        except Exception as e:
            logging.error(f"Execution fault: {e}")
            return ""

    def map_kernel(self) -> None:
        logging.info("Mapping Kernel parameters...")
        self.state.kernel_version = self._execute_safe(["uname", "-r"])

    def map_suid_sgid(self) -> None:
        logging.info("Hunting for high-sensitivity privilege boundary binaries...")
        target_dirs = ["/usr/bin", "/usr/sbin", "/bin", "/sbin", "/opt"]
        command = ["find"] + target_dirs + ["-perm", "-4000", "-type", "f"]
        
        raw_output = self._execute_safe(command)
        if raw_output:
            self.state.suid_binaries = raw_output.split('\n')

    def map_sudo_privs(self) -> None:
        logging.info("Evaluating privileged execution surface...")
        self.state.sudo_privileges = self._execute_safe(["sudo", "-l"])

    def execute_parallel_recon(self) -> Dict[str, Any]:
        recon_tasks = [self.map_kernel, self.map_suid_sgid, self.map_sudo_privs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(recon_tasks)) as executor:
            futures = [executor.submit(task) for task in recon_tasks]
            concurrent.futures.wait(futures)
        return asdict(self.state)

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    output_path = os.path.join(project_root, "data", "raw_state.json")

    engine = LinuxReconEngine(timeout=10)
    system_matrix = engine.execute_parallel_recon()
    
    with open(output_path, "w") as f:
        json.dump(system_matrix, f, indent=4)
    
    print(f"\n[+] Reconnaissance complete. Matrix saved to {output_path}")

