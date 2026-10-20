"""Source-inferred font proposals, resolved to explicitly available families.

Availability is limited to the project's verified candidate catalog, not a claim
that a family is the source's original font or that a target Office has it.
"""
import math
from .typography import font_status

ALIASES = {'microsoft yahei': '微软雅黑', '雅黑': '微软雅黑', '微软雅黑': '微软雅黑',
           'simhei': '黑体', 'simsun': '宋体', 'dengxian': '等线'}


def family_name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError('invalid font family')
    return ALIASES.get(value.strip().casefold(), value.strip())


def validate_font_proposal(text, *, required=False):
    if not required and 'font_family' not in text:
        return
    family_name(text.get('font_family'))
    family_name(text.get('font_latin', text['font_family']))
    confidence = text.get('font_confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('font confidence must be finite in [0,1]')
    if not isinstance(text.get('font_evidence'), str) or not 1 <= len(text['font_evidence'].strip()) <= 500:
        raise ValueError('font proposal requires visual evidence')


def resolve_font(text, available=None):
    validate_font_proposal(text, required=True)
    catalog = font_status()['families'] if available is None else available
    names = {family_name(name).casefold(): family_name(name) for name in catalog}
    resolved = {}
    for key, proposed in [('ea', text['font_family']), ('latin', text.get('font_latin', text['font_family']))]:
        canonical = family_name(proposed)
        if canonical.casefold() not in names:
            raise ValueError('font unavailable in verified catalog: ' + canonical)
        resolved[key] = names[canonical.casefold()]
    return resolved
