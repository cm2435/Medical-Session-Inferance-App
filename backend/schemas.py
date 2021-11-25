from ast import Str
from pydantic import BaseModel
from typing import Optional

class inferance_payload(BaseModel):
    user_name : Optional[str]
    session_number : Optional[int]
    session_note : str