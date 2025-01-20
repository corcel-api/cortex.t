from cortext import protocol
from cortext.base import BaseValidator
import bittensor as bt
import asyncio


class DumpValidator(BaseValidator):
    def __init__(self):
        super().__init__()

    async def start_epoch(self):
        print("Starting epoch")

    def set_weights(self):
        print("Setting weights")


async def main():

    validator = DumpValidator()

    target_axon = validator.metagraph.axons[1]
    print(target_axon)
    target_axon.ip = "127.0.0.1"
    target_axon.port = 8999

    response = await validator.dendrite.forward(
        axons=[target_axon],
        synapse=protocol.Credit(),
        streaming=False,
        timeout=16,
    )

    print(response)

    synapse = protocol.ChatStreamingProtocol(
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=100,
    )

    print(synapse)

    
    response = await validator.dendrite.forward(
        axons=[target_axon],
        synapse=synapse,
        streaming=True,
        timeout=16,
        deserialize=False,
    )
    response = response[0]
    async for chunk in response:
        print(chunk)
        print("---")

    print(chunk.completion)

    print(chunk.dendrite.process_time)


if __name__ == "__main__":
    asyncio.run(main())
