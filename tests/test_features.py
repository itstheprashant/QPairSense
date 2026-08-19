from app.services.feature_engineering import preprocess, pair_features

def test_preprocess_lowercases_and_removes_html():
    result = preprocess("I've <b>ALREADY</b> done!")
    assert "i have" in result
    assert "<b>" not in result

def test_pair_features_has_22_features():
    features = pair_features(
        "Where is the capital of India?",
        "Which city is the capital of India?"
    )
    assert len(features) == 22

def test_identical_questions_have_high_token_similarity():
    features = pair_features("How are you?", "How are you?")
    assert features[6] > 0
    assert features[21] >= 90
