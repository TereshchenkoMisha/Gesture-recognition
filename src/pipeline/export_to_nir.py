import os
import sys
import torch
import nir
from snntorch.export_nir import export_to_nir

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(current_dir)


from src.core.model.model import GestureSNN

device = torch.device("cpu")

# Loading of the trained model
model = GestureSNN(input_channels=2, num_classes=5).to(device)
model.load_state_dict(torch.load("best_emg_model.pth", map_location=device))
model.eval()

# Create a test input of the required form: [Time, Batch, Channels, H, W]
sample = torch.randn(20, 1, 2, 128, 128).to(device)

# Exporting to NIR
nir_graph = export_to_nir(model, sample, model_name="gesture_snn")

# Fixing None in node attributes
def sanitize_node_attrs(node):
    """Заменяет None на безопасное значение для сериализации."""
    for attr_name, attr_value in list(node.__dict__.items()):
        if attr_value is None:
            if 'shape' in attr_name:
                setattr(node, attr_name, [])
            else:
                setattr(node, attr_name, 0)
        elif isinstance(attr_value, dict):
            sanitize_dict(attr_value)
        elif hasattr(attr_value, '__dict__'):
            sanitize_node_attrs(attr_value)

def sanitize_dict(d):
    for k, v in d.items():
        if v is None:
            d[k] = []
        elif isinstance(v, dict):
            sanitize_dict(v)

for node_name, node in nir_graph.nodes.items():
    sanitize_node_attrs(node)

# Let's also clear the root graph just in case.
sanitize_dict(nir_graph.__dict__)

# Save the corrected graph
nir.write("gesture_snn.nir", nir_graph)
print("NIR экспорт завершён успешно!")