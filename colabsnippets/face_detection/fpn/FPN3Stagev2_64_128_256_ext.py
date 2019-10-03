from colabsnippets.face_detection.fpn.FPN3Stagev2Base import FPN3Stagev2Base


class FPN3Stagev2_64_128_256_ext(FPN3Stagev2Base):
  def __init__(self, name='fpn3stagev2_64_128_256_e', with_batch_norm=False):
    super().__init__(name=name, with_batch_norm=with_batch_norm, stage_filters=[64, 128, 256], channel_multiplier=1,
                     is_extended_first_layer=True)
