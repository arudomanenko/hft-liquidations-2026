from enum import Enum
import numpy as np


class ValidationErrorReason(Enum):
    HAS_NANs = 1
    HAS_INFs = 2
    HAS_FORWARD_LOOKING = 3
    INCORRECNT_TRADES_TO_TS_ALIGNMENT = 4
    OK = 5


def _validate_nans(data: np.ndarray) -> bool:
    data = np.asarray(data)
    return bool(not np.isnan(data).any())


def _validate_infs(data: np.ndarray) -> bool:
    data = np.asarray(data)
    return bool(not np.isinf(data).any())


def _validate_forward_looking(trades_ts: np.ndarray, max_used_ts: np.ndarray) -> bool:
    trades_ts = np.asarray(trades_ts)
    max_used_ts = np.asarray(max_used_ts)
    return bool(np.all(max_used_ts <= trades_ts))


def _validate_incorrect_aligments(trades_ts, feature_ts):
    trades_ts = np.asarray(trades_ts)
    feature_ts = np.asarray(feature_ts)
    return bool(np.all(trades_ts == feature_ts))


def validate(**data_to_validate) -> ValidationErrorReason:
    if not _validate_nans(data_to_validate["features"]):
        return ValidationErrorReason.HAS_NANs
    if not _validate_infs(data_to_validate["features"]):
        return ValidationErrorReason.HAS_INFs
    if not _validate_forward_looking(data_to_validate["trades_ts"], data_to_validate["max_used_ts"]):
        return ValidationErrorReason.HAS_FORWARD_LOOKING
    if not _validate_incorrect_aligments(data_to_validate["trades_ts"], data_to_validate["features_ts"]):
        return ValidationErrorReason.INCORRECNT_TRADES_TO_TS_ALIGNMENT
    return ValidationErrorReason.OK

