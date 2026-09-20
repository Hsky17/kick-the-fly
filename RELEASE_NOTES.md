# Kick the Fly 2.8.2: crash fix

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.8.1](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.8.1) is in this build.

## For players

**The game crashed when a second sugar pile, alcohol drop or bomb was used up.** Drop two sugar piles, let the fly eat
its way through the one that wasn't dropped first, and the game died with "The truth value of an array with more than
one element is ambiguous". The same crash was waiting behind the alcohol drops, the bombs, and the screen flashes and
damage popups. It's fixed, in both the 3D and the 2D game. Everything else is unchanged.

## For researchers

These effects live in lists of dicts whose values are numpy arrays, and they were removed with `list.remove()`, which
scans the list comparing with `==`. CPython short-circuits that on identity, so removing the *first* item worked and
hid the bug; anything later compared two dicts, `dict.__eq__` compared their arrays element-wise, and `bool()` on the
resulting array raised. They are now removed by identity through one helper, `kickthefly/game/kick_the_fly.py:drop_item`,
with `tests/test_drop_item.py` covering the helper and both games eating a second sugar pile.

Thanks to the player who sent the crash report, and whose machine (Nobara, Wayland, Radeon RX 9070 XT) turned out to
have nothing to do with it: the bug was in the game, and anyone with two sugar piles could hit it.
