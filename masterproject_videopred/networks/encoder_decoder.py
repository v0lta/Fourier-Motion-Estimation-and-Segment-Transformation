class EncoderDecoder:
    """Prediction model composed of an encoder and decoder."""
    def __init__(self, encoder, reconstructor=None, predictor=None):
        self._encoder = encoder
        self._reconstructor = reconstructor
        self._predictor = predictor

    def __call__(self, inputs):
        states = self._encoder(inputs)
        reconstruction = None
        prediction = None
        if self._reconstructor is not None:
            reconstruction = self._reconstructor(states, "reconstructor")
        if self._predictor is not None:
            prediction = self._predictor(states, "predictor")
        return (reconstruction, prediction)
