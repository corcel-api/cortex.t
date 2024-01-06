import requests
import websocket

from test_base import ActiveSubnetworkBaseTest

VALIDATOR_PORT = 8001
AXON_PORT = 8045


class Test(ActiveSubnetworkBaseTest):

    @classmethod
    def check_if_validator_is_up(cls):
        try:
            requests.get(f'http://localhost:{VALIDATOR_PORT}/', timeout=1)
        except requests.RequestException:
            return False
        return True

    @classmethod
    def check_if_miner_is_up(cls):
        try:
            websocket.create_connection(f'ws://localhost:{AXON_PORT}', timeout=1)
        except ConnectionRefusedError:
            return False
        except websocket.WebSocketBadStatusException:
            return True
        return True

    @classmethod
    def miner_path_and_args(cls) -> list[str]:
        return ['cortex-t-miner', '--netuid', '49', '--subtensor.network', 'test', '--wallet.name', 'miner',
                '--wallet.hotkey', 'default', '--axon.port', str(AXON_PORT)]

    @classmethod
    def validator_path_and_args(cls) -> list[str]:
        return ['cortex-t-validator', '--netuid', '49', '--subtensor.network', 'test', '--wallet.name',
                'validator', '--wallet.hotkey', 'default', '--http_port', str(VALIDATOR_PORT)]

    def test_text_validator(self):
        resp = requests.post(
            f'http://localhost:{VALIDATOR_PORT}/text-validator/',
            headers={'access-key': 'hello'},
            json={'1': 'please write a sentence using the word "cucumber"'},
            timeout=15,
        )
        resp.raise_for_status()
        assert "cucumber" in resp.text
        print(resp.text)
