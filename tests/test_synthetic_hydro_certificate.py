"""Independent correctness checks for the hydro certificate and continuous oracle."""
from fractions import Fraction
import numpy as np

from synthetic_hydro_certificate.model import Hydro, draw_paths, support, scenario_tree_oracle
from synthetic_hydro_certificate.certificate import CertificateEngine, bellman_values, generators, report_certificate
from synthetic_hydro_certificate.validation import audit_engines


def fixture():
    h = Hydro(prices=(1, 4, 2))
    fit = draw_paths([0.4,0.6,0.5],64,170)
    parameters = np.tile([0.3,0.6],(h.T,1))
    return h, [CertificateEngine(h,g,parameters) for g in generators(h,fit)]


def test_continuous_inner_matches_exact_integer_dp():
    h, engines = fixture()
    rows = audit_engines(engines,[21,40,34],64,tuple(range(2**h.T)))
    assert max(r['absolute_error'] for r in rows) < 1e-7
    assert any(r['mixed_integer'] for r in rows if r['generator']=='pwl')


def test_exact_continuation_has_pathwise_strong_duality():
    h=Hydro(prices=(1,4,2))
    p=[Fraction(2,5),Fraction(3,5),Fraction(1,2)]
    v=bellman_values(h,p,'oracle_audit_only')
    e=CertificateEngine(h,v,np.tile([0.3,0.6],(h.T,1)))
    values,_=e.inner_exact([4,6,5],10)
    exact=[Fraction(int(x),10*e.denominator) for x in values]
    assert all(x==v.values[0][h.initial_storage] for x in exact)
    lp=scenario_tree_oracle(h,p)
    assert abs(lp['value']-float(exact[0]))<1e-8


def test_affine_continuation_cancels_in_paired_gap():
    _,engines=fixture()
    a=engines[0].calibrate([21,40,34],64)
    b=engines[1].calibrate([21,40,34],64)
    np.testing.assert_allclose(a['raw'],b['raw'],atol=1e-10,rtol=0)
    assert not np.allclose(a['beta'],b['beta'])


def test_uniform_band_and_envelope_cover_fractional_queries():
    h,engines=fixture()
    rng=np.random.default_rng(919)
    from synthetic_hydro_certificate.certificate import exact_response
    for e in engines:
        c=e.calibrate([21,40,34],64)
        for t in range(h.T):
            for _ in range(25):
                s=Fraction(int(rng.integers(0,1001))*h.state_upper(t),1000)
                x=min(s,h.release_capacity)*Fraction(int(rng.integers(0,1001)),1000)
                difference=abs(exact_response(h,e.continuation,t,s,x,1)-exact_response(h,e.continuation,t,s,x,0))
                assert difference<=e.contrast_bounds[t]
        assert c['corrected'].min()>=0
        assert c['corrected'].max()<=c['envelope']
        result=report_certificate(c,np.ones(2**h.T,dtype=int))
        assert result['certificate']>=np.mean(c['corrected'])
