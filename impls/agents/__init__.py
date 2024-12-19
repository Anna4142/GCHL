from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.hsac import HSACAgent  # Hierarchical SAC
from agents.gchbc import GCHBCAgent  # Import GCHBC agent
from agents.gchiql import GCHIQLAgent

from agents.hgciql import HGCIQLAgent
from agents.hbc import HBCAgent
agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
   
    gchiql=GCHIQLAgent,
    hsac=HSACAgent,  # Hierarchical SAC
    gchbc=GCHBCAgent,  # Goal-Conditioned Hierarchical BC agent
    hgciql=HGCIQLAgent,
    hbc=HBCAgent
)
