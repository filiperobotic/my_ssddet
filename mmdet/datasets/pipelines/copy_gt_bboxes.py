from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class CopyGTBBoxesAsTrueBBoxes:
    def __call__(self, results):
        # Copia gt_bboxes para gt_true_bboxes
        # results['gt_true_bboxes'] = results['gt_bboxes'].copy()
        # Acessa os dados dentro do DataContainer e copia
        results['gt_true_bboxes'] = results['gt_bboxes'].data.copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'