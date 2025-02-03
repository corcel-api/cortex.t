from substrateinterface import Keypair
import time


def get_headers(keypair: Keypair):
    hotkey = keypair.public_key
    message = f"{hotkey}:{time.time()}"
    signature = f"0x{keypair.sign(message).hex()}"
    return {"Authorization": f"0x{signature}"}
