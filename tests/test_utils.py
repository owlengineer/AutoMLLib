import logging
from src.automllib.utils import merge_dicts

logger = logging.getLogger(__name__)


def test_merge_dicts():
    dst = {"A": 5, "B": 8, "C": 10}
    src = {"A": 10}
    merge_dicts(dst,src)
    assert dst == {"A": 10, "B": 8, "C": 10}