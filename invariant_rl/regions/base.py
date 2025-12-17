
class Region2D:
    def inside(self, x):
        raise NotImplementedError
    def sample_inside(self, N, **kw):
        raise NotImplementedError
    def sample_on_boundary(self, N, **kw):
        raise NotImplementedError
