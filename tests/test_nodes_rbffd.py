import numpy as np

from kernelpack import domain, geometry, nodes, rbffd


def test_poisson_sampler_and_geometry_clip():
    x2a, info2a = nodes.generate_poisson_nodes_in_box(0.075, [0, 0], [1, 1], seed=19, strip_count=5)
    x2b, _ = nodes.generate_poisson_nodes_in_box(0.075, [0, 0], [1, 1], seed=19, strip_count=5)
    assert np.array_equal(x2a, x2b)
    assert info2a["deterministic"]
    if x2a.shape[0] > 1:
        d = geometry.distance_matrix(x2a, x2a)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= 0.075 * (1 - 1e-10)

    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    curve = np.column_stack([np.cos(t), 0.7 * np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.05, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    gen = nodes.DomainNodeGenerator()
    gen.generate_interior_nodes_from_geometry(surface, 0.08, seed=29, strip_count=5)
    raw = gen.get_raw_poisson_interior_nodes()
    interior = gen.get_interior_nodes()
    assert interior.shape[0] < raw.shape[0]
    phi = surface.get_level_set().evaluate(interior)
    assert np.all(phi <= -0.08 + 1e-10)


def test_closed_curve_boundary_sampling_is_arc_length_uniform():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    curve = np.column_stack([np.cos(t), 0.7 * np.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.05, curve.shape[0])
    xb = surface.get_sample_sites()
    assert xb.shape[0] > 10
    seg_lens = np.linalg.norm(np.vstack([xb[1:], xb[:1]]) - xb, axis=1)
    assert np.all(seg_lens > 0)
    assert seg_lens.max() / seg_lens.min() < 1.5


def test_rbffd_laplacian():
    xg, yg = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), indexing="ij")
    x = np.column_stack([xg.ravel(), yg.ravel()])
    interior_mask = (np.abs(x[:, 0]) < 0.999) & (np.abs(x[:, 1]) < 0.999)
    active_rows = np.flatnonzero(interior_mask) + 1
    dd = domain.DomainDescriptor()
    dd.set_nodes(x, np.zeros((0, 2)), np.zeros((0, 2)))
    dd.set_sep_rad(0.5)
    dd.build_structs()
    sp = rbffd.StencilProperties(n=9, dim=2, ell=2, spline_degree=3, tree_mode="interior_boundary", point_set="interior_boundary")
    op = rbffd.OpProperties(record_stencils=True)
    f = x[:, 0] ** 2 + x[:, 1] ** 2
    fd_wls = rbffd.FDDiffOp(lambda: rbffd.WeightedLeastSquaresStencil())
    fd_wls.assemble_op(dd, "lap", sp, op, active_rows=active_rows)
    lwls = fd_wls.get_op()
    lap = lwls @ f
    assert np.all(np.abs(lap[active_rows - 1] - 4) < 1e-8)
