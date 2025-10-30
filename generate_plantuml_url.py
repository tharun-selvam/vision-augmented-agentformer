
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

[Camera Images] as CI
[Past Trajectories] as PT

component BEVDepth {
}

component AgentFormer {
}

[BEV Feature Map] as BFM
[Future Trajectories] as FT

CI -> BEVDepth
BEVDepth -> BFM
PT -> AgentFormer
BFM -> AgentFormer
AgentFormer -> FT

@enduml"""

encoded = plantuml_encode(plantuml_code)
print(f"http://www.plantuml.com/plantuml/png/{encoded}")
