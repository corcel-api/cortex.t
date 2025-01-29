from cortext import protocol
from neurons.validator import Validator
import asyncio


class DumpValidator(Validator):
    async def start_epoch(self):
        print("Starting epoch")

    def set_weights(self):
        print("Setting weights")


async def main():
    validator = DumpValidator()
    validator.miner_manager.step(
        [1, 0, 1],
        [1, 2, 3],
    )
    synapse = protocol.ChatStreamingProtocol(
        messages=[
            {
                "role": "user",
                "content": "Who are you?",
            }
        ],
        model="gpt-4o",
        temperature=0.0,
        max_tokens=4096,
        stream=True,
    )
    axon = validator.metagraph.axons[1]
    response = await validator.dendrite.forward(
        [axon],
        synapse,
        streaming=True,
        timeout=16,
        deserialize=False,
    )
    response = response[0]
    async for chunk in response:
        print(chunk)
    for _ in range(10):
        result = validator.miner_manager.consume(0.5, 5, 1)
        print(result)

    print(validator.miner_manager.weights)
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
