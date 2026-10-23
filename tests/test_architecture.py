from autoscale.env.metrics_gen import metric_names


def test_metric_generator_has_paper_metric_count():
    assert len(metric_names()) == 128
