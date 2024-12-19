from brainscore_vision import load_model, load_stimulus_set
from brainscore_vision.model_interface import BrainModel

model = load_model('alexnet')
model.start_recording(recording_target=BrainModel.RecordingTarget.V1, time_bins=[(100, 200)])
stimuli = load_stimulus_set('FreemanZiemba2013.aperture-public')  # load some images for the model to look at
neural_predictions = model.look_at(stimuli)
print(neural_predictions)

layer_names = model.layers
print(layer_names)

import matplotlib.pyplot as plt
plt.hist(neural_predictions.values.flatten(), bins=50)
plt.title('Activation Histogram')
plt.show()
