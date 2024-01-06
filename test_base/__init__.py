import abc
import logging
import os
import select
import subprocess
import time
import typing
from threading import Thread

import pytest


logger = logging.getLogger(__name__)


class ActiveSubnetworkBaseTest(abc.ABC):

    miner_process = None
    validator_process = None
    validator_stdout_thread: Thread | None = None
    validator_stderr_thread: Thread | None = None
    miner_stdout_thread: Thread | None = None
    miner_stderr_thread: Thread | None = None

    @classmethod
    @abc.abstractmethod
    def validator_path_and_args(cls) -> list[str]:
        """
        Return the path to validator start script and all arguments required to start it, e.g.
        `validator.py --opt1=8 --opt2=9`
        """
        ...

    @classmethod
    @abc.abstractmethod
    def miner_path_and_args(cls) -> list[str]:
        """
        Return the path to miner start script and all arguments required to start it, e.g.
        `miner.py --opt1=8 --opt2=9`
        """
        ...

    @classmethod
    @abc.abstractmethod
    def check_if_validator_is_up(cls) -> bool:
        """
        Check if the validator process is up and ready to accept requests. This can be achieved by making a network,
        reading logs etc.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def check_if_miner_is_up(cls) -> bool:
        """
        Check if the miner process is up and ready to accept requests. This can be achieved by making a network, reading
        logs etc.
        """
        ...

    @classmethod
    def make_log_reader(cls, io: typing.IO, name: str, process: subprocess.Popen):
        def read_logs():
            while True:
                r, _, _ = select.select([io], [], [], 1)
                if process.poll() is not None:
                    return
                if r:
                    line = io.readline()
                    if not line:
                        return
                    logger.info(f'{name}: {line.decode()[:-1]}')
        return read_logs

    @classmethod
    def start_process(cls, args):
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    @classmethod
    def wait_for_process_start(cls, process_name, probe_function, process: subprocess.Popen):
        for i in range(300):
            if probe_function():
                return
            if process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f'Process {process_name} did not start\n'
                           f'stdout={process.stdout.read()}\n\n'
                           f'stderr={process.stderr.read()}')

    @classmethod
    @pytest.fixture(autouse=True, scope="session")
    def start_validator_and_miner(cls):
        logger.info('Starting miner')
        cls.miner_process = cls.start_process(cls.miner_path_and_args())
        cls.miner_stdout_thread = Thread(target=cls.make_log_reader(cls.miner_process.stdout, 'miner stdout', cls.miner_process))
        cls.miner_stderr_thread = Thread(target=cls.make_log_reader(cls.miner_process.stderr, 'miner stderr', cls.miner_process))
        cls.miner_stdout_thread.start()
        cls.miner_stderr_thread.start()

        logger.info('Starting validator')
        cls.validator_process = cls.start_process(cls.validator_path_and_args())
        cls.validator_stdout_thread = Thread(target=cls.make_log_reader(cls.validator_process.stdout, 'validator stdout', cls.validator_process))
        cls.validator_stderr_thread = Thread(target=cls.make_log_reader(cls.validator_process.stderr, 'validator stderr', cls.validator_process))
        cls.validator_stdout_thread.start()
        cls.validator_stderr_thread.start()

        logger.info('Waiting for validator to start')
        cls.wait_for_process_start('validator', cls.check_if_validator_is_up, cls.validator_process)
        logger.info('Waiting for miner to start')
        cls.wait_for_process_start('miner', cls.check_if_miner_is_up, cls.miner_process)
        logger.info('Miner and validator started')
        yield
        if cls.miner_process:
            logger.info('Killing miner')
            os.killpg(os.getpgid(cls.miner_process.pid), 9)
        if cls.validator_process:
            logger.info('Killing validator')
            os.killpg(os.getpgid(cls.validator_process.pid), 9)
