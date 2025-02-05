from substrateinterface import Keypair
import time


def get_headers(keypair: Keypair):
    hotkey = keypair.ss58_address
    message = f"{hotkey}:{time.time()}"
    signature = f"0x{keypair.sign(message).hex()}"
    return {"signature": signature, "ss58_address": hotkey, "message": message}
