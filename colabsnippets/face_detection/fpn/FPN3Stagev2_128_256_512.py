from colabsnippets.face_detection.fpn.FPN3Stagev2Base import FPN3Stagev2Base


class FPN3Stagev2_128_256_512(FPN3Stagev2Base):
  def __init__(self, name='fpn3stagev2_128_256_512', with_batch_norm=False):
    super().__init__(name=name, with_batch_norm=with_batch_norm, stage_filters=[128, 256, 512], channel_multiplier=2)
