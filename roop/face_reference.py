from typing import Optional

from roop.typing import Face

FACE_REFERENCE = None


def get_face_reference() -> Optional[Face]: # type: ignore
    return FACE_REFERENCE


def set_face_reference(face: Face) -> None: # type: ignore
    global FACE_REFERENCE

    FACE_REFERENCE = face


def clear_face_reference() -> None:
    global FACE_REFERENCE

    FACE_REFERENCE = None
