# Project Manas

Monolingual classical literary Tibetan modeling. The current focus is on pretrained transformer models for:

- Monolingual tasks that are useful for teaching to read Tibetan, especially word segmentation, part-of-speech tagging and named entity recognition.
- Use as an encoder for the machine translation model.

## A note on configuration files

The configuration files for runnable code should be maintained to be the current champion methodology for whatever that code is doing. For training, the configs should contain the hyperparameters of the champion model. For inference, the configs should fix the example of how to best do whatever task.
