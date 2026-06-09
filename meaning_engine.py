meaning_rules = {
    ("GUN", "PERSON"): "GUNMAN",
    ("PERSON", "GUN"): "ARMED PERSON",

    ("RHYTHM", "PERSON"): "DANCER",
    ("DOREMIFASOLATI", "PERSON"): "MUSICIAN",

    ("SONG", "PERSON"): "SINGER",
    ("PERSON", "SONG"): "SINGER",

    ("MOVIE", "PERSON"): "ACTOR",
    ("PERSON", "MOVIE"): "ACTOR",

    ("TVSERIES", "PERSON"): "TV ACTOR",
    ("PERSON", "TVSERIES"): "TV ACTOR",

    ("GUN", "MOVIE"): "ACTION MOVIE",
    ("GUN", "TVSERIES"): "CRIME SERIES",

    ("SAD", "MOVIE"): "SAD MOVIE",
    ("HAPPY", "MOVIE"): "COMEDY MOVIE",

    ("ANIME", "MOVIE"): "ANIME MOVIE",
    ("ANIME", "TVSERIES"): "ANIME SERIES",

    ("DRUNK", "PERSON"): "DRUNK PERSON",
    ("PERSON", "DRUNK"): "DRUNK PERSON",

    ("HAPPY", "PERSON"): "HAPPY PERSON",
    ("PERSON", "HAPPY"): "HAPPY PERSON",

    ("SAD", "PERSON"): "SAD PERSON",
    ("PERSON", "SAD"): "SAD PERSON",

    ("ANIME", "SONG"): "ANIME SONG",
    ("SONG", "ANIME"): "ANIME SONG",

    ("BIRD", "SONG"): "BIRD SONG",
    ("SONG", "BIRD"): "BIRD SONG",

    ("SONG", "RHYTHM"): "MUSIC",
    ("RHYTHM", "DOREMIFASOLATI"): "MELODY",

    ("PERSON", "PERSON"): "PEOPLE",

    ("PERSON", "GUN", "MOVIE"): "ACTION HERO",
    ("PERSON", "SONG", "RHYTHM"): "SINGER / PERFORMER",
    ("PERSON", "DOREMIFASOLATI", "SONG"): "MUSICIAN",
    ("PERSON", "ANIME", "MOVIE"): "ANIME FAN",
    ("PERSON", "HAPPY", "SONG"): "HAPPY SINGER",
    ("PERSON", "SAD", "SONG"): "SAD SINGER",
    ("GUN", "PERSON", "MOVIE"): "GUNMAN IN MOVIE",
    ("MOVIE", "SONG", "RHYTHM"): "MUSICAL MOVIE",
    ("PERSON", "ANIME", "SONG"): "ANIME PERFORMER",
    ("ANIME", "PERSON", "SONG"): "ANIME SINGER",
    ("PERSON", "MOVIE", "SONG"): "MOVIE STAR",
    ("PERSON", "HAPPY", "MOVIE"): "COMEDY ACTOR",
    ("PERSON", "SAD", "MOVIE"): "TRAGEDY ACTOR",
    ("PERSON", "BIRD", "SONG"): "BIRD FAN",

    ("MOVIE", "PERSON", "GUN"): "ACTION STAR",
    ("SONG", "PERSON", "MOVIE"): "MUSIC STAR",
    ("ANIME", "PERSON"): "ANIME FAN",
    ("BIRD", "PERSON"): "BIRD WATCHER",
    ("HAPPY", "SONG"): "HAPPY SONG",
    ("SAD", "SONG"): "SAD SONG",
}


def guess_meaning(sign_stack):
    signs = list(sign_stack)

    # Check last 3 signs first
    if len(signs) >= 3:
        last_three = tuple(signs[-3:])
        if last_three in meaning_rules:
            return meaning_rules[last_three]

    # Then check last 2 signs
    if len(signs) >= 2:
        last_two = tuple(signs[-2:])
        if last_two in meaning_rules:
            return meaning_rules[last_two]

    # Check last 1 sign
    if len(signs) >= 1:
        return signs[-1]

    return "NO MEANING YET"