from .models import GlobalLocalDualBranch

def make_model(model_name):
    model_class = globals()[model_name]
    model = model_class()
    return model