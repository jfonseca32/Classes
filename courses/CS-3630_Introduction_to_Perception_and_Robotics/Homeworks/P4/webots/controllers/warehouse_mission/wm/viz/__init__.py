"""Webots presentation layer of the Project 4 warehouse mission (staff code).

``wm.viz`` turns the ``StepResult`` stream of :class:`wm.core.Mission` into
the cinematic Webots show: the 3D scene animation, cameras, sound, the HUD,
and the "Robot Vision" floor hologram. Students never need to read it.

* ``app.py``: the presentation state machine and the controller main loop
  (entry point: :func:`wm.viz.app.main`, called by ``warehouse_mission.py``).
* ``scene.py``: every Webots node the show animates (robot, lights, beacons,
  docks, forklifts, markers, parcels, obstacles).
* ``camera.py``: camera modes, scripted flyovers, smoothing.
* ``audio.py``: sound effects and optional text-to-speech.
* ``util.py``: small math helpers (look-at rotations, easing, smoothing).
* ``hud.py``: the supervisor labels (the only module that calls ``setLabel``).
* ``hologram.py`` / ``font5x7.py``: NumPy renderers of the floor hologram and
  the minimap (no Webots import; unit-testable offline).

This file imports nothing on purpose: ``hologram.py`` and ``font5x7.py`` must
stay importable without Webots, and importing them runs this file first.
"""
