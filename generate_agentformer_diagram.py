
import zlib
import base64
import sys

def plantuml_encode(plantuml_text):
    """Encode plantuml text for URL.
    Copied from https://github.com/dougn/python-plantuml/blob/master/plantuml.py
    """
    plantuml_alphabet = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    base64_alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    b64_to_plantuml = bytes.maketrans(base64_alphabet, plantuml_alphabet)

    zlibbed_str = zlib.compress(plantuml_text.encode("utf-8"))
    compressed_string = zlibbed_str[2:-4]
    return base64.b64encode(compressed_string).translate(b64_to_plantuml).decode("utf-8")


plantuml_code = """@startuml
skinparam componentStyle rectangle

component "Past Trajectories" as Past
component "Future Trajectories (GT)" as FutureGT
component "AgentFormer Encoder" as Encoder
component "AgentFormer Decoder" as Decoder
component "CVAE" as CVAE
component "Predicted Trajectories" as Pred

Past -> Encoder
Encoder -> CVAE
FutureGT -> CVAE
CVAE -> Decoder
Decoder -> Pred

@enduml"""

encoded = plantuml_encode(plantuml_code)
print(f"http://www.plantuml.com/plantuml/png/{encoded}")
