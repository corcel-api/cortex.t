import asyncio
import os
import sys
from unittest import mock

import bittensor
import pytest
import torch

from cortex_t.validators import weight_setter
from cortex_t.validators.validator import main, validator_app

hotkeys = os.environ.get('CORTEXT_MINER_ADDITIONAL_WHITELIST_VALIDATOR_KEYS', '').split(',')

hotkeys += ['mock'] * (7 - len(hotkeys))

synthetic_question = "tell me why aint nothing but a heartbreak"

synthetic_resp1 = """
The phrase "ain't nothing but a heartbreak" is a line from the song "I Want It That Way" by the Backstreet Boys, which was released in 1999. The song is about the complexities of a relationship and the pain of being apart from the person you love. The line suggests that the situation they are singing about causes nothing but emotional pain and heartache.

In a broader sense, the phrase can be used to describe any situation that causes significant emotional distress, particularly in the context of romantic relationships. It's a way of expressing that the primary outcome of a situation is heartbreak.
"""

synthetic_resp2 = synthetic_resp1 + ' And that\'s why.'

synthetic_resp3 = """
The phrase "ain't nothing but a heartbreak" is a lyric from the song "I Want It That Way" by the Backstreet Boys, a popular boy band from the late 1990s and early 2000s. The song was released in 1999 as part of their album "Millennium" and quickly became one of their signature hits.

The line is part of the chorus:

"Tell me why
Ain't nothin' but a heartache
Tell me why
Ain't nothin' but a mistake
Tell me why
I never wanna hear you say
I want it that way"

In the context of the song, the phrase expresses the pain and frustration of a romantic relationship that is causing heartache. The song's lyrics deal with themes of love, regret, and misunderstanding between partners. The phrase "ain't nothing but a heartbreak" suggests that the relationship is causing nothing but emotional pain, emphasizing the depth of the narrator's distress.
"""

organic_question = "What is black thunder?"

organic_question_1 = organic_question + ' 1'
organic_question_2 = organic_question + ' 2'

organic_answer_1 = """
Black Thunder could refer to different things depending on the context. Here are a few possibilities:

Amusement Park: Black Thunder could refer to an amusement park. There's a famous water theme park in Tamil Nadu, India, called Black Thunder, known for its water rides and entertainment attractions.
Military Operations: Sometimes, military operations or exercises are given code names. "Black Thunder" might be the name of a specific military operation conducted by a particular country's armed forces.
Film or Media: There might be movies, books, or other media with the title "Black Thunder." It could be a novel, film, or series with a plot related to action, adventure, or a specific theme.
Nickname or Alias: It might also be a nickname or alias used by an individual or a group for various purposes. It could be in reference to someone's personality, actions, or a particular event.
Without additional context, it's challenging to pinpoint the exact reference to "Black Thunder." If you have more details or a specific context in mind, I could provide more accurate information.
"""

organic_answer_2 = organic_answer_1 + " that would be it."

organic_answer_3 = """
"Yellow lightning" typically refers to a type of lightning that appears to have a yellowish or amber hue during a thunderstorm. Lightning usually appears as a bright flash or streak in the sky during a thunderstorm due to the discharge of electricity between clouds or between a cloud and the ground.

The color of lightning can vary depending on various factors, such as atmospheric conditions, the presence of particles or gases in the air, or the distance between the observer and the lightning strike. Lightning often appears as white or bluish-white, but it can also exhibit different colors like yellow, orange, or even red.

The yellowish or amber hue in lightning might be caused by the scattering of light through a greater distance due to atmospheric conditions or the presence of particles. However, the exact reason for the yellow coloration in lightning can vary and is still an area of study among meteorologists and atmospheric scientists.
"""


def feed_mock_data(text_validator):
    text_validator.feed_mock_data(
        {
            synthetic_question + ' 1': [synthetic_resp1, synthetic_resp2],
            synthetic_question + ' 2': [synthetic_resp1, synthetic_resp3],
            synthetic_question + ' 3': [synthetic_resp2, synthetic_resp1],
            synthetic_question + ' 4': [synthetic_resp2, synthetic_resp3],
            synthetic_question + ' 5': [synthetic_resp3, synthetic_resp1],
            synthetic_question + ' 6': [synthetic_resp3, synthetic_resp2],
            organic_question_1: [organic_answer_1, organic_answer_2],
            organic_question_2: [organic_answer_2, organic_answer_3],
        },
        [synthetic_question + f' {i}' for i in range(1, 7)]
    )


async def assert_weights_update(set_weights_mock: mock.Mock, expected_weights: torch.tensor):
    previous_calls = len(set_weights_mock.call_args_list)
    for _ in range(400):
        await asyncio.sleep(0.25)
        if len(set_weights_mock.call_args_list) > previous_calls:
            assert len(set_weights_mock.call_args_list) == previous_calls + 1
            assert all(set_weights_mock.call_args_list[-1].kwargs['weights'] == expected_weights)
            break
    else:
        raise ValueError('set_weights_mock not called')


expected_scores_after_one_iteration = torch.tensor([1.0, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408,
                                                    0.3333333432674408, 0.3333333432674408, 1.0])


@pytest.mark.asyncio
async def test_synthetic_and_organic(aiohttp_client):
    with (mock.patch.object(bittensor.subtensor, 'set_weights') as set_weights_mock,
          mock.patch.object(bittensor.metagraph, 'hotkeys', new=hotkeys),
          mock.patch.object(weight_setter, 'SYNTHETIC_SCORING_LOOP_SLEEP', 0.1)):
        sys.argv = ['validator.py', '--netuid', '49', '--subtensor.network', 'test', '--wallet.name', 'validator',
                    '--wallet.hotkey', 'default']
        main(run_aio_app=False, test=True)
        feed_mock_data(validator_app.weight_setter.text_vali)

        await assert_weights_update(set_weights_mock, expected_scores_after_one_iteration)

        validator_app.weight_setter.total_scores = torch.zeros(7)
        validator_app.weight_setter.moving_average_scores = None
        feed_mock_data(validator_app.weight_setter.text_vali)

        await assert_weights_update(set_weights_mock, expected_scores_after_one_iteration / 2)

        validator_app.weight_setter.total_scores = torch.zeros(7)
        validator_app.weight_setter.moving_average_scores = None
        feed_mock_data(validator_app.weight_setter.text_vali)

        client = await aiohttp_client(validator_app)

        resp = await client.post('/text-validator/', headers={'access-key': 'hello'}, json={'4': organic_question_1})
        resp_content = (await resp.content.read()).decode()
        assert resp_content == organic_answer_1

        resp = await client.post('/text-validator/', headers={'access-key': 'hello'}, json={'5': organic_question_2})
        resp_content = (await resp.content.read()).decode()
        assert resp_content == organic_answer_2

        await assert_weights_update(
            set_weights_mock,
            torch.tensor([0.3333333432674408, 0.111111119389534, 0.111111119389534, 0.111111119389534,
                          0.1388888955116272,  # this one was asked a question and answered correctly
                          0.111111119389534,  # this one was asked a question and answered incorrectly
                          0.3333333432674408,
                          ])
        )



