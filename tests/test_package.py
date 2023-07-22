import sys
import importlib.metadata

import dynamic_routing_analysis


def test_import():
    assert dynamic_routing_analysis in sys.modules.values()
    assert dir(dynamic_routing_analysis)

def test_version():
    assert importlib.metadata.version('dynamic_routing_analysis')
    