"""Persian spec-key -> stable ASCII slug mappings.

``trim_specs.item_slug`` and ``trim_spec_groups.group_slug`` are primary-key
components, so these mappings must never change once data exists. Adding a new
key is safe; editing an existing one orphans rows.

Any key not listed here still round-trips safely: :func:`numbers.slugify_fa`
falls back to ``unk_<sha1[:10]>``, which is deterministic, and ``deep audit``
reports every ``unk_`` slug so the mapping can be completed deliberately rather
than guessed.

Keys are matched *after* ``normalize_text``, so Arabic/Persian character
variants and stray whitespace are already folded.
"""

from __future__ import annotations

from typing import Final

#: The 13 specification groups returned by /nws/api/CarReview/getspecification.
SPEC_GROUP_SLUGS: Final[dict[str, str]] = {
    "وضعیت در بازار ایران": "market_status",
    "مشخصات فنی": "technical",
    "عملکرد فنی": "performance",
    "ابعاد و اندازه ها": "dimensions",
    "سیستم‌ کمکی راننده": "driver_assist",
    "سیستم کمکی راننده": "driver_assist",
    "رانندگی هوشمند": "smart_driving",
    "سیستم‌های ترمز": "brakes",
    "سیستم های ترمز": "brakes",
    "سیستم‌ روشنایی": "lighting",
    "سیستم روشنایی": "lighting",
    "فرمان": "steering",
    "مالتی مدیا": "multimedia",
    "ایمنی کابین": "cabin_safety",
    "رفاهی کابین": "cabin_comfort",
    "آینه، شیشه‌ و اتاق": "mirrors_glass_body",
    "آینه، شیشه و اتاق": "mirrors_glass_body",
}

#: Individual specification items. Extended from real responses; incomplete by
#: design -- unmapped keys become ``unk_<hash>`` and are surfaced by the audit.
SPEC_KEY_SLUGS: Final[dict[str, str]] = {
    # market status
    "سال های موجود": "years_available",
    "سال‌های موجود": "years_available",
    # technical
    "محور محرک": "drive_shaft",
    "گیربکس": "transmission",
    "پیشرانه": "engine",
    "حجم موتور": "engine_displacement",
    "نوع سوخت": "fuel_type",
    # performance
    "قدرت": "power",
    "گشتاور": "torque",
    "شتاب": "acceleration",
    "سرعت": "top_speed",
    "حداکثر سرعت": "top_speed",
    "مصرف ترکیبی": "fuel_consumption_combined",
    "مصرف سوخت ترکیبی": "fuel_consumption_combined",
    "مصرف در شهر": "fuel_consumption_city",
    "مصرف در جاده": "fuel_consumption_highway",
    # dimensions
    "نوع بدنه": "body_type",
    "نوع شاسی": "chassis_type",
    "نسل": "generation",
    "طول": "length",
    "عرض": "width",
    "ارتفاع": "height",
    "فاصله محوری": "wheelbase",
    "وزن": "weight",
    "حجم باک": "fuel_tank_capacity",
    "حجم صندوق": "boot_capacity",
    "چرخ جلو": "front_tyre",
    "چرخ عقب": "rear_tyre",
    "رینگ": "wheel_rim",
    # driver assistance
    "کروز کنترل": "cruise_control",
    "کروز کنترل تطبیقی": "adaptive_cruise_control",
    "دستیار حرکت در سراشیبی": "hill_descent_assist",
    "دستیار حرکت در سربالایی": "hill_start_assist",
    "کنترل کشش": "traction_control",
    "کنترل پایداری": "stability_control",
    "سیستم جلوگیری از واژگونی": "rollover_prevention",
    "اتو پارک": "auto_park",
    "سیستم توقف و حرکت خودکار": "auto_hold",
    # smart driving
    "سنسور پارک عقب": "rear_parking_sensor",
    "سنسور پارک جلو": "front_parking_sensor",
    "سنسور باران": "rain_sensor",
    "سنسور فشار باد لاستیک": "tyre_pressure_monitoring",
    "هشدار نقطه کور": "blind_spot_warning",
    "هشدار خروج از خط": "lane_departure_warning",
    "دستیار حفظ خط": "lane_keep_assist",
    "هشدار برخورد جلو": "forward_collision_warning",
    "هشدار عبور عرضی عقب": "rear_cross_traffic_alert",
    "تشخیص علائم راهنمایی": "traffic_sign_recognition",
    "دستیار ترافیک": "traffic_jam_assist",
    # brakes
    "توزیع الکترونیکی نیروی ترمز": "ebd",
    "ترمز ضد قفل": "abs",
    "دستیار ترمز": "brake_assist",
    "ترمز پارک برقی": "electronic_parking_brake",
    "ترمز اضطراری خودکار": "autonomous_emergency_braking",
    # lighting
    "چراغ روزانه": "daytime_running_lights",
    "دی لایت": "daytime_running_lights",
    "چراغ خودکار": "auto_headlights",
    "اتولایت": "auto_headlights",
    "مه‌شکن جلو": "front_fog_lights",
    "مه شکن جلو": "front_fog_lights",
    "مه‌شکن عقب": "rear_fog_lights",
    "مه شکن عقب": "rear_fog_lights",
    "چراغ جلو": "headlight_type",
    # steering
    "نیروی کمکی فرمان": "power_steering",
    "دکمه‌های کنترلی فرمان": "steering_wheel_controls",
    "دکمه های کنترلی فرمان": "steering_wheel_controls",
    "فرمان حساس به سرعت": "speed_sensitive_steering",
    "پدل شیفتر": "paddle_shifters",
    # multimedia
    "سیستم صوتی": "audio_system",
    "مانیتور": "monitor",
    "دوربین عقب": "rear_camera",
    "دوربین ۳۶۰ درجه": "camera_360",
    "اتصالات": "connectivity",
    # cabin safety
    "مجموع ایربگ‌ها": "airbag_count",
    "مجموع ایربگ ها": "airbag_count",
    "ایربگ راننده": "driver_airbag",
    "ایربگ سرنشین": "passenger_airbag",
    "ایربگ پرده‌ای": "curtain_airbag",
    "ایربگ جانبی": "side_airbag",
    "ایزوفیکس": "isofix",
    "قفل کودک": "child_lock",
    "ایموبیلایزر": "immobilizer",
    # cabin comfort
    "تهویه": "climate_control",
    "سیستم تهویه": "climate_control",
    "صندلی راننده": "driver_seat",
    "صندلی برقی": "power_seat",
    "روکش صندلی": "seat_upholstery",
    "گرمکن صندلی": "heated_seats",
    "استارت بدون کلید": "keyless_start",
    "ورود بدون کلید": "keyless_entry",
    # mirrors / glass / body
    "آینه تاشو برقی": "power_folding_mirrors",
    "سانروف": "sunroof",
    "پانوراما": "panoramic_roof",
    "شیشه بالابر برقی": "power_windows",
    # EV-only
    "ظرفیت باتری": "battery_capacity",
    "برد الکتریکی": "all_electric_range",
    "زمان شارژ": "charging_time",
    # --- added from the first live run (reported as unk_ by the audit) -------
    "نسل (کد اتاق)": "generation_code",
    "سیستم تعلیق": "suspension_system",
    "حالت رانندگی": "driving_modes",
    "سنسور پایش راننده": "driver_monitoring_sensor",
    "ارتفاع نور اتومات": "auto_headlight_leveling",
    "نوربالا اتومات": "auto_high_beam",
    "چراغ شور": "headlight_washer",
    "چراغ چرخشی": "cornering_lights",
    "چراغ مه‌شکن جلو": "front_fog_lights",
    "چراغ مه‌شکن عقب": "rear_fog_lights",
    "دی‌لایت": "daytime_running_lights",
    "نیروی کمکی": "power_steering",
    "دکمه‌های کنترلی": "steering_wheel_controls",
    "حساس به سرعت": "speed_sensitive_steering",
    "گرمکن فرمان": "heated_steering_wheel",
    # "other features" appears in several groups; the primary key includes the
    # group slug, so one shared item slug is correct and unambiguous.
    "سایر ویژگی‌ها": "other_features",
    "سایر ویژگی‌ روشنایی": "other_lighting_features",
    "سایر ویژگی‌ ترمز": "other_brake_features",
    "سایر ویژگی‌ فرمان": "other_steering_features",
    "سایر ویژگی‌ مالتی مدیا": "other_multimedia_features",
    "سایر ویژگی‌ ایمنی": "other_safety_features",
    # --- second pass: the remaining keys the first live run reported ---------
    # cabin safety
    "ایربگ سرنشین جلو": "front_passenger_airbag",
    "ایربگ جانبی جلو": "front_side_airbag",
    "ایربگ جانبی عقب": "rear_side_airbag",
    "ایربگ زانویی راننده": "driver_knee_airbag",
    "ایربگ زانویی سرنشین جلو": "passenger_knee_airbag",
    "ایربگ بین راننده و سرنشین": "front_centre_airbag",
    "پشت‌سری فعال": "active_head_restraints",
    # cabin comfort
    "کول باکس": "cool_box",
    "ماساژور صندلی": "massaging_seats",
    "سردکن صندلی": "ventilated_seats",
    "ویژگی صندلی عقب": "rear_seat_features",
    "صندلی سرنشین جلو": "front_passenger_seat",
    "ایزوفیکس صندلی": "isofix",
    "استارت دکمه‌ای": "push_button_start",
    "صندوق پران اتومات": "automatic_boot_release",
    "صندوق برقی با جک": "power_tailgate",
    "سایر ویژگی‌ کابین": "other_cabin_features",
    # multimedia
    "نویگیشن": "navigation",
    "هدآپ": "head_up_display",
    "دوربین جانبی": "side_camera",
    "دوربین شب": "night_vision_camera",
    "قابلیت پشتیبانی": "smartphone_integration",
    "نوع صفحه کیلومتر": "instrument_cluster_type",
    # mirrors / glass / body
    "سقف پانوراما": "panoramic_roof",
    "آینه عقب الکتروکرومیک": "auto_dimming_rear_mirror",
    "آینه جانبی تاشو برقی": "power_folding_mirrors",
    "آینه جانبی با تنظیم برقی": "power_adjustable_mirrors",
    "سایر ویژگی‌ ظاهری": "other_exterior_features",
    # Keys whose only ASCII content is a number; without an explicit mapping the
    # fallback would slug them as "12" / "360".
    "خروجی 12 ولت": "power_outlet_12v",
    "دوربین 360": "camera_360",
    # --- driver-assist and brake systems -------------------------------------
    # Bama writes these as "<Persian name> (<ACRONYM>)". The ASCII fallback used
    # to strip the Persian and land on the acronym by accident; these mappings
    # make the (correct) slug explicit and stable.
    "ترمز ضدقفل (ABS)": "abs",
    "توزیع نیرو ترمز (EBD)": "ebd",
    "ترمز با نیروی کمکی (BA)": "ba",
    "ترمز پارک برقی (EPB)": "epb",
    "ترمز اضطراری هوشمند (AEB)": "aeb",
    "کروز کنترل هوشمند (ACC)": "acc",
    "رادار نقطه کور (BSD)": "bsd",
    "کنترل پایداری (ESC)": "esc",
    "رادار تصادف جلو (FCW)": "fcw",
    "کنترل سرعت در سرازیری (HDA)": "hda",
    "کنترل ایستایی در سربالایی (HSA)": "hsa",
    "رادار تغییرلاین (LDW)": "ldw",
    "رادار ماندن در لاین (LKA)": "lka",
    "رادار تصادف عقب (RCTA)": "rcta",
    "ضد واژگونی (ROP)": "rop",
    "کنترل کشش (TCS)": "tcs",
    "رادار حرکت در ترافیک (TJA)": "tja",
    "سنسور فشار باد تایر (TPMS)": "tpms",
    "رادار تابلوخوان (TSR)": "tsr",
}
