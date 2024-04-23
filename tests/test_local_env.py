from xtal2txt.local_env import LocalEnvAnalyzer


def test_structure_to_local_env_string(get_incus2):
    lea = LocalEnvAnalyzer()
    incus2 = get_incus2
    string = lea.structure_to_local_env_string(incus2)
    expected = "I-42d\nS2- (8d) [Cu]S([In])([In])[Cu]\nCu+ (4a) [S][Cu]([S])([S])[S]\nIn3+ (4b) [S][In]([S])[S].[S]"
    assert string == expected
