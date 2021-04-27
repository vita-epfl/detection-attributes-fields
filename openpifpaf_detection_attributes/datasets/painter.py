import openpifpaf


class BoxPainter(openpifpaf.show.DetectionPainter):
    """Painter for bounding boxes of detected instances.

    Args:
        xy_scale (float): Scale factor for display.
    """

    def __init__(self, *, xy_scale: float = 1.0):
        super().__init__(xy_scale=xy_scale)


    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        assert 'center' in ann.attributes
        assert 'width' in ann.attributes
        assert 'height' in ann.attributes
        anndet = openpifpaf.annotation.AnnotationDet([]).set(0, 0.,
            [ann.attributes['center'][0]-.5*ann.attributes['width'],
             ann.attributes['center'][1]-.5*ann.attributes['height'],
             ann.attributes['width'], ann.attributes['height']])

        if text is None:
            text = ann.object_type.name
        if subtext is None:
            if getattr(ann, 'id', None): # ground truth annotation
                subtext = ann.id
            elif 'confidence' in ann.attributes: # prediction
                subtext = '{:.0%}'.format(ann.attributes['confidence'])

        super().annotation(ax, anndet, color=color, text=text, subtext=subtext)
