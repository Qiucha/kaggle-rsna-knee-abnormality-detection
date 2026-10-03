import pytest

from contrastive_learning.acquisition import Acquisition, is_fat_suppressed, recover


# --- the two traps the prior work named, which silently invert the answer ----------- #

def test_underscores_do_not_hide_water_excitation():
    # Underscore is a word character, so \bwe\b never fires inside t2_de3d_we_tra
    # unless separators are normalised first.
    assert is_fat_suppressed("t2_de3d_we_tra", None, None) is True
    assert is_fat_suppressed("t2 de3d we tra", None, None) is True


def test_spatial_saturation_is_not_fat_saturation():
    # GE writes SAT_GEMS for *spatial* saturation; a substring test for "SAT" would
    # mark this non-suppressed series as suppressed.
    assert is_fat_suppressed("Ax PD FSE", None, "SAT_GEMS") is False
    assert is_fat_suppressed("Ax PD FSE", None, "FS\\SAT_GEMS") is True


@pytest.mark.parametrize("description", ["pd_tse_fs_cor_320", "PDW_SPAIR", "Cor STIR", "t1 tirm"])
def test_known_suppressed_descriptions(description):
    assert is_fat_suppressed(description, None, None) is True


@pytest.mark.parametrize("description", ["pd_tse_cor_320", "Sag T1 SE", "t2 tse tra"])
def test_known_unsuppressed_descriptions(description):
    assert is_fat_suppressed(description, None, None) is False


# --- weighting ---------------------------------------------------------------------- #

def test_description_beats_timing_when_it_names_a_weighting():
    # TR/TE here would say T1; the radiographer says PD, and that is the direct statement.
    result = recover(description="pd_tse_fs_cor_320", repetition_time=500, echo_time=10)
    assert (result.weighting, result.source) == ("PD", "description")


def test_pd_wins_over_t2_because_pd_sequences_often_name_both():
    assert recover(description="pd_t2_tse_cor").weighting == "PD"


def test_timing_decides_when_the_description_is_silent():
    assert recover(description="cor 320", repetition_time=500).weighting == "T1"
    assert recover(description="cor 320", repetition_time=3000, echo_time=80).weighting == "T2"
    assert recover(description="cor 320", repetition_time=3000, echo_time=30).weighting == "PD"


def test_gradient_echo_is_its_own_class_not_forced_into_the_spin_echo_rule():
    result = recover(description="3d cor", scanning_sequence="GR", repetition_time=20, echo_time=5)
    assert (result.weighting, result.source) == ("GRE", "timing")


def test_missing_timing_is_unknown_rather_than_guessed():
    assert recover(description="cor 320").weighting == "UNK"


def test_fluid_sensitivity_follows_weighting_not_fat_suppression():
    # The whole point of the recovery: these are independent axes.
    assert recover(description="pd_tse_cor").fluid_sensitive is True
    assert recover(description="t2 tse cor").fluid_sensitive is True
    assert recover(description="Sag T1 SE").fluid_sensitive is False
    assert recover(description="Sag T1 SE fs").fat_suppressed is True
    assert recover(description="Sag T1 SE fs").fluid_sensitive is False


def test_result_is_hashable_so_slots_can_be_keyed_on_it():
    assert isinstance(recover(description="pd_tse_fs_cor"), Acquisition)
    assert len({recover(description="pd_tse_fs_cor"), recover(description="pd_tse_fs_cor")}) == 1
