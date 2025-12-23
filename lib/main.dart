// Road Health V2 - main.dart (single-file reference implementation)
//
// Key upgrades vs V1
// 1) Separate DetectionEvent entity (reliable trigger -> evidence linkage)
// 2) Mount-agnostic inertial features (magnitude, windowing, cooldown, speed gating)
// 3) Correct distance tracking (current -> new position)
// 4) Debounce / peak-window detection to reduce repeated triggers
// 5) Vision verifier abstraction (drop-in custom TFLite model later). ML Kit remains as fallback only.
//
// Notes
// - iOS background sensing has stricter constraints than Android. This code is optimized for Android.
// - For production, split into files, add tests, and implement a custom on-device CV model.


import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:share_plus/share_plus.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

// ============================================================================
// DESIGN SYSTEM
// ============================================================================

class RHSColors {
  static const Color primary = Color(0xFF0D7377);
  static const Color primaryLight = Color(0xFF14A3A8);
  static const Color primaryDark = Color(0xFF095557);
  static const Color success = Color(0xFF2ECC71);
  static const Color warning = Color(0xFFF39C12);
  static const Color danger = Color(0xFFE74C3C);
  static const Color dangerDark = Color(0xFFC0392B);
  static const Color bgPrimary = Color(0xFFF7F9FC);
  static const Color bgSecondary = Color(0xFFEEF2F7);
  static const Color bgCard = Color(0xFFFFFFFF);
  static const Color textPrimary = Color(0xFF2D3436);
  static const Color textSecondary = Color(0xFF636E72);
  static const Color textMuted = Color(0xFFB2BEC3);

  static const Color severityCritical = Color(0xFFE74C3C);
  static const Color severityHigh = Color(0xFFF39C12);
  static const Color severityMedium = Color(0xFFF1C40F);
  static const Color severityLow = Color(0xFF3498DB);

  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft, end: Alignment.bottomRight,
    colors: [primary, primaryLight],
  );

  static const LinearGradient dangerGradient = LinearGradient(
    begin: Alignment.topLeft, end: Alignment.bottomRight,
    colors: [danger, dangerDark],
  );

  static Color severityColor(double severity) {
    if (severity >= 75) return severityCritical;
    if (severity >= 50) return severityHigh;
    if (severity >= 25) return severityMedium;
    return severityLow;
  }

  static String severityLabel(double severity) {
    if (severity >= 75) return 'Critical';
    if (severity >= 50) return 'High';
    if (severity >= 25) return 'Medium';
    return 'Low';
  }
}

// ============================================================================
// DOMAIN ENTITIES
// ============================================================================

enum DefectType { pothole, speedBump, roughRoad, unknown }

extension DefectTypeExtension on DefectType {
  String get displayName {
    switch (this) {
      case DefectType.pothole: return 'Pothole';
      case DefectType.speedBump: return 'Speed Bump';
      case DefectType.roughRoad: return 'Rough Road';
      case DefectType.unknown: return 'Unknown Defect';
    }
  }

  IconData get icon {
    switch (this) {
      case DefectType.pothole: return Icons.warning_amber_rounded;
      case DefectType.speedBump: return Icons.speed;
      case DefectType.roughRoad: return Icons.terrain;
      case DefectType.unknown: return Icons.help_outline;
    }
  }
}

class DetectionEvent {
  DetectionEvent({
    required this.id, required this.timestamp, required this.latitude,
    required this.longitude, required this.speedKmh, required this.severityIndex,
    required this.rawPeak, required this.rawP2P, required this.modelConfidence,
    required this.type, this.photoPath, this.visionLabel, this.visionConfidence,
    this.roadName, this.ward, this.repeatCount = 1,
  });

  final String id;
  final DateTime timestamp;
  final double latitude, longitude, speedKmh, severityIndex, rawPeak, rawP2P, modelConfidence;
  DefectType type;
  String? photoPath, visionLabel, roadName, ward;
  double? visionConfidence;
  int repeatCount;

  LatLng get latLng => LatLng(latitude, longitude);

  double get priorityScore {
    final severityWeight = severityIndex / 100.0;
    final repeatWeight = min(repeatCount / 5.0, 1.0);
    return (severityWeight * 0.5 + repeatWeight * 0.3 + modelConfidence * 0.2) * 100;
  }

  Map<String, dynamic> toJson() => {
    'id': id, 'timestamp': timestamp.toIso8601String(),
    'latitude': latitude, 'longitude': longitude, 'speed_kmh': speedKmh,
    'severity_index': severityIndex, 'raw_peak': rawPeak, 'raw_p2p': rawP2P,
    'inertial_confidence': modelConfidence, 'type': type.name,
    'photo': photoPath ?? '', 'vision_label': visionLabel ?? '',
    'vision_confidence': visionConfidence ?? '', 'road_name': roadName ?? '',
    'ward': ward ?? '', 'repeat_count': repeatCount, 'priority_score': priorityScore,
  };
}

class DefectCluster {
  DefectCluster({required this.id, required this.centroid, required this.events, this.roadName, this.ward});

  final String id;
  final LatLng centroid;
  final List<DetectionEvent> events;
  String? roadName, ward;

  double get avgSeverity => events.isEmpty ? 0 : events.map((e) => e.severityIndex).reduce((a, b) => a + b) / events.length;
  double get maxSeverity => events.isEmpty ? 0 : events.map((e) => e.severityIndex).reduce(max);
  int get detectionCount => events.length;

  DateTime? get firstSeen => events.isEmpty ? null : events.map((e) => e.timestamp).reduce((a, b) => a.isBefore(b) ? a : b);
  DateTime? get lastSeen => events.isEmpty ? null : events.map((e) => e.timestamp).reduce((a, b) => a.isAfter(b) ? a : b);

  DefectType get dominantType {
    if (events.isEmpty) return DefectType.unknown;
    final counts = <DefectType, int>{};
    for (final e in events) counts[e.type] = (counts[e.type] ?? 0) + 1;
    return counts.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  double get priorityScore => (avgSeverity / 100.0 * 0.6 + min(detectionCount / 10.0, 1.0) * 0.4) * 100;
}

// ============================================================================
// DETECTION ENGINE
// ============================================================================

class _AccelSample {
  _AccelSample(this.t, this.ax, this.ay, this.az);
  final DateTime t;
  final double ax, ay, az;
  double get magnitude => sqrt(ax * ax + ay * ay + az * az);
}

class DetectionEngine {
  static const int windowMs = 450;
  static const int minWindowSamples = 8;
  static const Duration cooldown = Duration(milliseconds: 1500);
  static const double minSpeedKmh = 5.0;
  static const double maxReasonableSpeedKmh = 160.0;

  static double thresholdForSpeed(double speedKmh) => 14.0 + 0.10 * speedKmh.clamp(0.0, maxReasonableSpeedKmh);

  static double severityIndex(double peakMag, double p2p, double speedKmh) {
    final denom = max(8.0, speedKmh);
    return ((peakMag / denom) * 100.0 + (p2p / denom) * 40.0).clamp(0.0, 100.0);
  }

  final List<_AccelSample> _buf = [];
  DateTime? _lastTrigger;
  bool _armed = false;
  DateTime? _armStart;

  DetectionEvent? ingest({required _AccelSample sample, required Position? position, required double speedKmh}) {
    if (position == null || speedKmh < minSpeedKmh) return null;
    if (_lastTrigger != null && sample.t.difference(_lastTrigger!) < cooldown) return null;

    _buf.add(sample);
    final cutoff = sample.t.subtract(const Duration(milliseconds: windowMs));
    while (_buf.isNotEmpty && _buf.first.t.isBefore(cutoff)) _buf.removeAt(0);
    if (_buf.length < minWindowSamples) return null;

    double peak = 0.0, minMag = double.infinity, maxMag = 0.0;
    for (final s in _buf) {
      final m = s.magnitude;
      peak = max(peak, m); minMag = min(minMag, m); maxMag = max(maxMag, m);
    }
    final p2p = (maxMag - minMag).abs();
    final thr = thresholdForSpeed(speedKmh);

    if (!_armed && peak > thr) { _armed = true; _armStart = sample.t; return null; }

    if (_armed) {
      final armedAge = sample.t.difference(_armStart!);
      if (armedAge.inMilliseconds >= 180) {
        final confidence = _inertialConfidence(peak, p2p, thr);
        if (confidence >= 0.75) {
          _armed = false; _armStart = null; _lastTrigger = sample.t;
          return DetectionEvent(
            id: '${sample.t.millisecondsSinceEpoch}-${(position.latitude * 100000).round() ^ (position.longitude * 100000).round()}',
            timestamp: sample.t, latitude: position.latitude, longitude: position.longitude,
            speedKmh: speedKmh, severityIndex: severityIndex(peak, p2p, speedKmh),
            rawPeak: peak, rawP2P: p2p, modelConfidence: confidence,
            type: _coarseType(peak, p2p, speedKmh),
          );
        } else if (armedAge.inMilliseconds >= 350) { _armed = false; _armStart = null; }
      }
    }
    return null;
  }

  double _inertialConfidence(double peak, double p2p, double thr) =>
    (0.65 * ((peak - thr).clamp(0.0, 20.0) / 20.0) + 0.35 * (p2p.clamp(0.0, 10.0) / 10.0)).clamp(0.0, 1.0);

  DefectType _coarseType(double peak, double p2p, double speedKmh) {
    if (speedKmh >= 12 && speedKmh <= 40 && p2p >= 5.5 && peak >= 20) return DefectType.speedBump;
    if (peak >= 22 && p2p >= 6.0) return DefectType.pothole;
    if (peak >= 18) return DefectType.roughRoad;
    return DefectType.unknown;
  }

  void reset() { _buf.clear(); _armed = false; _armStart = null; _lastTrigger = null; }
}

// ============================================================================
// VISION VERIFIER
// ============================================================================

abstract class VisionVerifier {
  Future<VisionResult?> verify(String imagePath);
  void close();
}

class VisionResult {
  VisionResult(this.label, this.confidence, this.refinedType);
  final String label;
  final double confidence;
  final DefectType refinedType;
}

class MlKitVisionVerifier implements VisionVerifier {
  MlKitVisionVerifier({double confidenceThreshold = 0.55})
    : _labeler = ImageLabeler(options: ImageLabelerOptions(confidenceThreshold: confidenceThreshold));
  final ImageLabeler _labeler;

  @override
  Future<VisionResult?> verify(String imagePath) async {
    try {
      final labels = await _labeler.processImage(InputImage.fromFilePath(imagePath));
      if (labels.isEmpty) return null;
      labels.sort((a, b) => b.confidence.compareTo(a.confidence));
      final top = labels.first;
      final txt = top.label.toLowerCase();
      DefectType refined = DefectType.unknown;
      if (txt.contains('speed') || txt.contains('bump')) refined = DefectType.speedBump;
      if (txt.contains('road') || txt.contains('asphalt')) refined = DefectType.roughRoad;
      return VisionResult(top.label, top.confidence, refined);
    } catch (e) { debugPrint('Vision verify error: $e'); return null; }
  }

  @override
  void close() => _labeler.close();
}

// ============================================================================
// DEMO DATA GENERATOR
// ============================================================================

class DemoDataGenerator {
  static final Random _rand = Random(42);
  static const double _baseLat = -26.2041, _baseLon = 28.0473;

  static final List<String> _roadNames = [
    'Commissioner St', 'Market St', 'Pritchard St', 'Jeppe St', 'Bree St',
    'Main Reef Rd', 'Empire Rd', 'Jan Smuts Ave', 'Oxford Rd', 'Rivonia Rd',
    'William Nicol Dr', 'Sandton Dr', 'Grayston Dr', 'Katherine St',
  ];

  static final List<String> _wards = [
    'Ward 60 - Inner City', 'Ward 64 - Braamfontein', 'Ward 87 - Sandton',
    'Ward 89 - Rosebank', 'Ward 103 - Randburg', 'Ward 112 - Fourways',
  ];

  static List<DetectionEvent> generateDemoEvents(int count) {
    final events = <DetectionEvent>[];
    final types = [DefectType.pothole, DefectType.pothole, DefectType.pothole, DefectType.roughRoad, DefectType.speedBump, DefectType.unknown];

    for (int i = 0; i < count; i++) {
      final lat = _baseLat + (_rand.nextDouble() - 0.5) * 0.15;
      final lon = _baseLon + (_rand.nextDouble() - 0.5) * 0.15;
      final severity = 20 + _rand.nextDouble() * 80;
      final daysAgo = _rand.nextInt(30);
      final timestamp = DateTime.now().subtract(Duration(days: daysAgo, hours: _rand.nextInt(24)));

      events.add(DetectionEvent(
        id: 'demo-$i-${timestamp.millisecondsSinceEpoch}',
        timestamp: timestamp, latitude: lat, longitude: lon,
        speedKmh: 15 + _rand.nextDouble() * 50, severityIndex: severity,
        rawPeak: 15 + severity * 0.2, rawP2P: 5 + severity * 0.1,
        modelConfidence: 0.75 + _rand.nextDouble() * 0.25,
        type: types[_rand.nextInt(types.length)],
        roadName: _roadNames[_rand.nextInt(_roadNames.length)],
        ward: _wards[_rand.nextInt(_wards.length)],
        repeatCount: 1 + _rand.nextInt(8),
      ));
    }
    return events;
  }

  static List<DefectCluster> clusterEvents(List<DetectionEvent> events, {double radiusMeters = 50}) {
    if (events.isEmpty) return [];
    final clusters = <DefectCluster>[];
    final used = <int>{};
    const distance = Distance();

    for (int i = 0; i < events.length; i++) {
      if (used.contains(i)) continue;
      final seed = events[i];
      final clusterEvents = <DetectionEvent>[seed];
      used.add(i);

      for (int j = i + 1; j < events.length; j++) {
        if (used.contains(j)) continue;
        if (distance.as(LengthUnit.Meter, seed.latLng, events[j].latLng) <= radiusMeters) {
          clusterEvents.add(events[j]); used.add(j);
        }
      }

      final avgLat = clusterEvents.map((e) => e.latitude).reduce((a, b) => a + b) / clusterEvents.length;
      final avgLon = clusterEvents.map((e) => e.longitude).reduce((a, b) => a + b) / clusterEvents.length;

      clusters.add(DefectCluster(
        id: 'cluster-${clusters.length}', centroid: LatLng(avgLat, avgLon),
        events: clusterEvents, roadName: clusterEvents.first.roadName, ward: clusterEvents.first.ward,
      ));
    }
    return clusters;
  }
}

// ============================================================================
// ROAD HEALTH STATE
// ============================================================================

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(statusBarColor: Colors.transparent, statusBarIconBrightness: Brightness.dark));
  try { cameras = await availableCameras(); } catch (e) { debugPrint('Camera init error: $e'); }
  runApp(const RoadHealthApp());
}

class RoadHealthApp extends StatelessWidget {
  const RoadHealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Health Score', debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: RHSColors.primary, brightness: Brightness.light),
        scaffoldBackgroundColor: RHSColors.bgPrimary,
        textTheme: GoogleFonts.dmSansTextTheme(),
        cardTheme: const CardThemeData(elevation: 0, color: RHSColors.bgCard),
      ),
      home: const MainNavigationPage(),
    );
  }
}

class MainNavigationPage extends StatefulWidget {
  const MainNavigationPage({super.key});
  @override State<MainNavigationPage> createState() => _MainNavigationPageState();
}

class _MainNavigationPageState extends State<MainNavigationPage> {
  int _currentIndex = 0;
  final RoadHealthState _appState = RoadHealthState();

  @override void initState() { super.initState(); _appState.initialize(); }
  @override void dispose() { _appState.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          MonitorScreen(appState: _appState),
          MapScreen(appState: _appState),
          WorklistScreen(appState: _appState),
          SettingsScreen(appState: _appState),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(color: Colors.white, boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20, offset: const Offset(0, -4))]),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(0, Icons.sensors, Icons.sensors_outlined, 'Monitor'),
                _buildNavItem(1, Icons.map, Icons.map_outlined, 'Map'),
                _buildNavItem(2, Icons.assignment, Icons.assignment_outlined, 'Worklist'),
                _buildNavItem(3, Icons.settings, Icons.settings_outlined, 'Settings'),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(int index, IconData activeIcon, IconData inactiveIcon, String label) {
    final isActive = _currentIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _currentIndex = index),
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(color: isActive ? RHSColors.primary.withOpacity(0.1) : Colors.transparent, borderRadius: BorderRadius.circular(12)),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(isActive ? activeIcon : inactiveIcon, color: isActive ? RHSColors.primary : RHSColors.textMuted, size: 24),
          const SizedBox(height: 4),
          Text(label, style: GoogleFonts.dmSans(fontSize: 11, fontWeight: isActive ? FontWeight.w600 : FontWeight.w400, color: isActive ? RHSColors.primary : RHSColors.textMuted)),
        ]),
      ),
    );
  }
}

class RoadHealthState extends ChangeNotifier {
  bool isLogging = false, autoModeEnabled = false, isStarting = false, demoModeEnabled = false;
  double currentImpact = 0.0, maxImpact = 0.0;
  int totalSamples = 0, detections = 0, photosCaptured = 0;
  DateTime? sessionStartTime;
  Duration sessionDuration = Duration.zero;
  int lifetimeDetections = 0;
  double lifetimeKmMapped = 0.0;
  Position? currentPosition;
  double currentSpeedKmh = 0.0;
  DateTime? lastMovementTime;
  bool audioEnabled = true, hapticEnabled = true, cameraEnabled = true, evidenceCaptureEnabled = true;

  StreamSubscription<UserAccelerometerEvent>? _accelSub;
  Timer? _gpsTimer, _flushTimer, _autoStopTimer, _sessionTimer;
  DateTime? _lastUiUpdate;
  final DetectionEngine _engine = DetectionEngine();
  CameraController? cameraController;
  bool _isCapturing = false;
  final AudioPlayer _audioPlayer = AudioPlayer();
  VisionVerifier? _vision;
  final List<Map<String, dynamic>> _telemetryBatch = [];
  final List<DetectionEvent> _events = [];
  List<DefectCluster> _clusters = [];

  List<DetectionEvent> get events => List.unmodifiable(_events);
  List<DefectCluster> get clusters => List.unmodifiable(_clusters);

  Function(String)? onError, onSuccess;

  static const double autoStartSpeedThreshold = 10.0;
  static const Duration autoStopIdleDuration = Duration(minutes: 3);
  static const Duration uiUpdateInterval = Duration(milliseconds: 150);

  DefectType? worklistTypeFilter;
  String? worklistWardFilter;
  String worklistSortBy = 'priority';

  List<DefectCluster> get filteredClusters {
    var result = List<DefectCluster>.from(_clusters);
    if (worklistTypeFilter != null) result = result.where((c) => c.dominantType == worklistTypeFilter).toList();
    if (worklistWardFilter != null && worklistWardFilter!.isNotEmpty) result = result.where((c) => c.ward == worklistWardFilter).toList();
    switch (worklistSortBy) {
      case 'severity': result.sort((a, b) => b.maxSeverity.compareTo(a.maxSeverity)); break;
      case 'date': result.sort((a, b) => (b.lastSeen ?? DateTime(2000)).compareTo(a.lastSeen ?? DateTime(2000))); break;
      default: result.sort((a, b) => b.priorityScore.compareTo(a.priorityScore));
    }
    return result;
  }

  Set<String> get availableWards => _clusters.map((c) => c.ward).whereType<String>().toSet();

  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    lifetimeDetections = prefs.getInt('lifetime_detections') ?? 0;
    lifetimeKmMapped = prefs.getDouble('lifetime_km') ?? 0.0;
    if (cameras.isNotEmpty) {
      try {
        cameraController = CameraController(cameras.first, ResolutionPreset.medium, enableAudio: false);
        await cameraController!.initialize();
      } catch (e) { cameraEnabled = false; }
    } else { cameraEnabled = false; }
    _vision = MlKitVisionVerifier();
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(channelId: 'rhs_channel', channelName: 'Road Health Monitoring', channelDescription: 'Active road condition monitoring', channelImportance: NotificationChannelImportance.LOW, priority: NotificationPriority.LOW),
      iosNotificationOptions: const IOSNotificationOptions(showNotification: true, playSound: false),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(),
        autoRunOnBoot: false,
        allowWakeLock: true,
        allowWifiLock: true,
      ),
    );
    notifyListeners();
  }

  void enableDemoMode() {
    demoModeEnabled = true;
    _events.clear();
    _events.addAll(DemoDataGenerator.generateDemoEvents(75));
    _clusters = DemoDataGenerator.clusterEvents(_events);
    detections = _events.length;
    lifetimeDetections = _events.length;
    lifetimeKmMapped = 45.3;
    notifyListeners();
    onSuccess?.call('Demo mode enabled with ${_events.length} sample detections');
  }

  void disableDemoMode() {
    demoModeEnabled = false;
    _events.clear();
    _clusters.clear();
    detections = 0;
    notifyListeners();
    onSuccess?.call('Demo mode disabled');
  }

  Future<bool> requestPermissions() async {
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) { onError?.call('Location permission required'); return false; }
    }
    if (permission == LocationPermission.deniedForever) { onError?.call('Location permission denied forever'); return false; }
    if (Platform.isAndroid) {
      final notificationStatus = await Permission.notification.request();
      if (!notificationStatus.isGranted) { onError?.call('Notification permission required'); return false; }
    }
    if (cameraEnabled && evidenceCaptureEnabled) {
      final cameraStatus = await Permission.camera.request();
      if (!cameraStatus.isGranted) { onError?.call('Camera permission denied'); evidenceCaptureEnabled = false; }
    }
    return true;
  }

  Future<void> startLogging() async {
    if (isLogging || isStarting) return;
    isStarting = true; notifyListeners();
    try {
      if (!await requestPermissions()) return;
      if (await FlutterForegroundTask.isRunningService) await FlutterForegroundTask.restartService();
      else await FlutterForegroundTask.startService(notificationTitle: 'Road Health Active', notificationText: 'Monitoring road conditions...');

      isLogging = true; totalSamples = 0; detections = 0; photosCaptured = 0; maxImpact = 0.0;
      sessionStartTime = DateTime.now(); sessionDuration = Duration.zero;
      _telemetryBatch.clear();
      if (!demoModeEnabled) { _events.clear(); _clusters.clear(); }
      _engine.reset();

      _accelSub = userAccelerometerEventStream(samplingPeriod: const Duration(milliseconds: 20)).listen(_onAccelEvent);
      _gpsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _updateGPS());
      _flushTimer = Timer.periodic(const Duration(seconds: 20), (_) => _flushToDisk());
      _sessionTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (sessionStartTime != null) { sessionDuration = DateTime.now().difference(sessionStartTime!); notifyListeners(); }
      });
      if (autoModeEnabled) { lastMovementTime = DateTime.now(); _autoStopTimer = Timer.periodic(const Duration(seconds: 10), (_) => _checkAutoStop()); }
      onSuccess?.call('Monitoring started');
    } catch (e) { onError?.call('Failed to start: $e'); }
    finally { isStarting = false; notifyListeners(); }
  }

  Future<void> stopLogging() async {
    if (!isLogging) return;
    await _accelSub?.cancel(); _accelSub = null;
    _gpsTimer?.cancel(); _gpsTimer = null;
    _flushTimer?.cancel(); _flushTimer = null;
    _autoStopTimer?.cancel(); _autoStopTimer = null;
    _sessionTimer?.cancel(); _sessionTimer = null;
    await _flushToDisk();
    await FlutterForegroundTask.stopService();
    lifetimeDetections += detections;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('lifetime_detections', lifetimeDetections);
    await prefs.setDouble('lifetime_km', lifetimeKmMapped);
    _clusters = DemoDataGenerator.clusterEvents(_events);
    isLogging = false; notifyListeners();
    onSuccess?.call('Session complete: $detections detections');
  }

  void toggleAutoMode() {
    autoModeEnabled = !autoModeEnabled; notifyListeners();
    if (autoModeEnabled && !isLogging) {
      _gpsTimer = Timer.periodic(const Duration(seconds: 2), (_) => _updateGPS());
      onSuccess?.call('Auto mode ON');
    } else if (!autoModeEnabled && !isLogging) {
      _gpsTimer?.cancel(); _gpsTimer = null;
      onSuccess?.call('Manual mode enabled');
    }
  }

  void _onAccelEvent(UserAccelerometerEvent e) {
    final now = DateTime.now();
    final sample = _AccelSample(now, e.x, e.y, e.z);
    final mag = sample.magnitude;
    currentImpact = mag;
    if (mag > maxImpact) maxImpact = mag;

    if (_lastUiUpdate == null || now.difference(_lastUiUpdate!) >= uiUpdateInterval) { _lastUiUpdate = now; notifyListeners(); }
    if (currentPosition != null && totalSamples % 5 == 0) {
      _telemetryBatch.add({'timestamp': now.toIso8601String(), 'latitude': currentPosition!.latitude, 'longitude': currentPosition!.longitude, 'speed_kmh': currentSpeedKmh, 'impact_mag': mag});
    }
    totalSamples++;
    final event = _engine.ingest(sample: sample, position: currentPosition, speedKmh: currentSpeedKmh);
    if (event != null) _handleDetection(event);
  }

  Future<void> _handleDetection(DetectionEvent event) async {
    detections++;
    if (hapticEnabled) HapticFeedback.heavyImpact();
    if (audioEnabled) try { await _audioPlayer.play(AssetSource('sounds/beep.mp3')); } catch (_) { if (hapticEnabled) HapticFeedback.mediumImpact(); }
    _events.add(event);
    _clusters = DemoDataGenerator.clusterEvents(_events);
    notifyListeners();
    if (cameraEnabled && evidenceCaptureEnabled && cameraController != null && cameraController!.value.isInitialized) await _captureEvidenceFor(event);
  }

  Future<void> _captureEvidenceFor(DetectionEvent event) async {
    if (_isCapturing) return; _isCapturing = true;
    try {
      final dir = await getApplicationDocumentsDirectory();
      final imagePath = '${dir.path}/rhs_event_${event.id}_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final pic = await cameraController!.takePicture();
      await File(pic.path).copy(imagePath);
      event.photoPath = imagePath; photosCaptured++;
      if (_vision != null) {
        final r = await _vision!.verify(imagePath);
        if (r != null) { event.visionLabel = r.label; event.visionConfidence = r.confidence; }
      }
      notifyListeners();
    } catch (e) { debugPrint('Evidence capture error: $e'); }
    finally { _isCapturing = false; }
  }

  Future<void> _updateGPS() async {
    try {
      final pos = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
      final newSpeedKmh = (pos.speed * 3.6).isFinite ? pos.speed * 3.6 : 0.0;
      if (isLogging && currentPosition != null) {
        final d = Geolocator.distanceBetween(currentPosition!.latitude, currentPosition!.longitude, pos.latitude, pos.longitude);
        if (d > 5 && d < 1500) lifetimeKmMapped += d / 1000.0;
      }
      currentPosition = pos; currentSpeedKmh = newSpeedKmh;
      if (autoModeEnabled && !isLogging && !isStarting && currentSpeedKmh > autoStartSpeedThreshold) { if (await requestPermissions()) await startLogging(); }
      if (isLogging && currentSpeedKmh > 1.0) lastMovementTime = DateTime.now();
    } catch (e) { debugPrint('GPS error: $e'); }
  }

  void _checkAutoStop() {
    if (!isLogging || !autoModeEnabled || lastMovementTime == null) return;
    if (DateTime.now().difference(lastMovementTime!) >= autoStopIdleDuration) stopLogging();
  }

  Future<void> _flushToDisk() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];
      if (_telemetryBatch.isNotEmpty) {
        final path = '${dir.path}/rhs_telemetry_$date.csv';
        final file = File(path);
        final sb = StringBuffer();
        if (!await file.exists()) sb.writeln('timestamp,latitude,longitude,speed_kmh,impact_mag');
        for (final r in _telemetryBatch) sb.writeln('${r['timestamp']},${r['latitude']},${r['longitude']},${r['speed_kmh']},${r['impact_mag']}');
        await file.writeAsString(sb.toString(), mode: FileMode.append);
        _telemetryBatch.clear();
      }
      if (_events.isNotEmpty) {
        final evPath = '${dir.path}/rhs_events_$date.csv';
        final evFile = File(evPath);
        final sb = StringBuffer();
        if (!await evFile.exists()) sb.writeln('id,timestamp,latitude,longitude,speed_kmh,severity_index,type,road_name,ward,priority_score');
        for (final ev in _events) sb.writeln('${ev.id},${ev.timestamp.toIso8601String()},${ev.latitude},${ev.longitude},${ev.speedKmh},${ev.severityIndex},${ev.type.name},${ev.roadName ?? ''},${ev.ward ?? ''},${ev.priorityScore}');
        await evFile.writeAsString(sb.toString(), mode: FileMode.write);

        final features = _events.map((e) => {'type': 'Feature', 'properties': e.toJson(), 'geometry': {'type': 'Point', 'coordinates': [e.longitude, e.latitude]}}).toList();
        final geoPath = '${dir.path}/rhs_events_$date.geojson';
        await File(geoPath).writeAsString(jsonEncode({'type': 'FeatureCollection', 'name': 'RoadHealth Events $date', 'features': features}));
      }
    } catch (e) { debugPrint('Flush error: $e'); }
  }

  Future<void> exportSessionFiles() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];
      final files = <XFile>[];
      for (final name in ['rhs_telemetry_$date.csv', 'rhs_events_$date.csv', 'rhs_events_$date.geojson']) {
        final f = File('${dir.path}/$name');
        if (await f.exists()) files.add(XFile(f.path));
      }
      if (files.isEmpty) { onError?.call('No exported files found'); return; }
      await Share.shareXFiles(files, subject: 'Road Health Export $date');
      onSuccess?.call('Export shared');
    } catch (e) { onError?.call('Export failed: $e'); }
  }

  @override
  void dispose() { stopLogging(); _audioPlayer.dispose(); cameraController?.dispose(); _vision?.close(); super.dispose(); }

  String get formattedSessionDuration {
    final m = sessionDuration.inMinutes.toString().padLeft(2, '0');
    final s = (sessionDuration.inSeconds % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}

// ============================================================================
// MONITOR SCREEN
// ============================================================================

class MonitorScreen extends StatefulWidget {
  final RoadHealthState appState;
  const MonitorScreen({super.key, required this.appState});
  @override State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  @override void initState() { super.initState(); widget.appState.addListener(_onStateChange); widget.appState.onError = _showError; widget.appState.onSuccess = _showSuccess; }
  @override void dispose() { widget.appState.removeListener(_onStateChange); super.dispose(); }
  void _onStateChange() { if (mounted) setState(() {}); }
  void _showError(String msg) { if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg), backgroundColor: RHSColors.danger)); }
  void _showSuccess(String msg) { if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg), backgroundColor: RHSColors.success)); }

  @override
  Widget build(BuildContext context) {
    final state = widget.appState;
    return Scaffold(
      body: SafeArea(
        child: ListView(padding: const EdgeInsets.all(20), children: [
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            Row(children: [
              Image.asset('assets/images/rhs_logo.png', height: 40),
              const SizedBox(width: 12),
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Road Health', style: GoogleFonts.dmSans(fontSize: 24, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
                Text('Municipal road intelligence', style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary)),
              ]),
            ]),
            Container(
              decoration: BoxDecoration(color: state.demoModeEnabled ? RHSColors.warning.withOpacity(0.15) : RHSColors.bgSecondary, borderRadius: BorderRadius.circular(8), border: state.demoModeEnabled ? Border.all(color: RHSColors.warning, width: 1.5) : null),
              child: IconButton(onPressed: () => state.demoModeEnabled ? state.disableDemoMode() : state.enableDemoMode(), icon: Icon(Icons.science_outlined, color: state.demoModeEnabled ? RHSColors.warning : RHSColors.textMuted), tooltip: state.demoModeEnabled ? 'Disable Demo Mode' : 'Enable Demo Mode'),
            ),
          ]),
          if (state.demoModeEnabled)
            Container(margin: const EdgeInsets.only(top: 12), padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8), decoration: BoxDecoration(color: RHSColors.warning.withOpacity(0.15), borderRadius: BorderRadius.circular(8), border: Border.all(color: RHSColors.warning.withOpacity(0.3))),
              child: Row(children: [const Icon(Icons.info_outline, size: 18, color: RHSColors.warning), const SizedBox(width: 8), Expanded(child: Text('Demo mode: Showing sample Johannesburg data', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.warning, fontWeight: FontWeight.w500)))]),
            ),
          const SizedBox(height: 24),
          Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(gradient: state.isLogging ? RHSColors.primaryGradient : null, color: state.isLogging ? null : RHSColors.bgSecondary, borderRadius: BorderRadius.circular(16)),
            child: Column(children: [
              Icon(state.isLogging ? Icons.sensors : Icons.pause_circle, size: 48, color: state.isLogging ? Colors.white : RHSColors.textMuted),
              const SizedBox(height: 12),
              Text(state.isLogging ? 'MONITORING ACTIVE' : 'PAUSED', style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700, color: state.isLogging ? Colors.white : RHSColors.textMuted)),
              if (state.isLogging) ...[const SizedBox(height: 8), Text('${state.formattedSessionDuration}  •  ${state.currentSpeedKmh.toStringAsFixed(0)} km/h', style: GoogleFonts.dmSans(color: Colors.white.withOpacity(0.9)))],
            ]),
          ),
          const SizedBox(height: 20),
          Row(children: [
            Expanded(child: _buildStatCard('Impact', state.currentImpact.toStringAsFixed(1), state.currentImpact > DetectionEngine.thresholdForSpeed(state.currentSpeedKmh) ? RHSColors.danger : RHSColors.primary)),
            const SizedBox(width: 12),
            Expanded(child: _buildStatCard('Max Impact', state.maxImpact.toStringAsFixed(1), RHSColors.warning)),
          ]),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: _buildStatCard('Detections', '${state.detections}', RHSColors.danger)),
            const SizedBox(width: 12),
            Expanded(child: _buildStatCard('Photos', '${state.photosCaptured}', RHSColors.success)),
          ]),
          const SizedBox(height: 20),
          GestureDetector(onTap: state.toggleAutoMode,
            child: Container(padding: const EdgeInsets.all(16), decoration: BoxDecoration(color: state.autoModeEnabled ? RHSColors.primary.withOpacity(0.1) : Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: state.autoModeEnabled ? RHSColors.primary : Colors.transparent)),
              child: Row(children: [
                Icon(Icons.directions_car, color: state.autoModeEnabled ? RHSColors.primary : RHSColors.textMuted),
                const SizedBox(width: 12),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Smart Auto Mode', style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)),
                  Text(state.autoModeEnabled ? 'Starts at 10 km/h, stops after 3 min idle' : 'Enable hands-free operation', style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary)),
                ])),
                Switch(value: state.autoModeEnabled, onChanged: (_) => state.toggleAutoMode(), activeColor: RHSColors.primary),
              ]),
            ),
          ),
          const SizedBox(height: 16),
          Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(gradient: RHSColors.primaryGradient, borderRadius: BorderRadius.circular(16)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('COVERAGE STATS', style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white.withOpacity(0.8), letterSpacing: 1)),
              const SizedBox(height: 12),
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                _buildImpactStat('${state.lifetimeDetections}', 'Total Detections'),
                _buildImpactStat('${state.lifetimeKmMapped.toStringAsFixed(1)} km', 'Roads Mapped'),
                _buildImpactStat('${state.clusters.length}', 'Clusters'),
              ]),
            ]),
          ),
          const SizedBox(height: 24),
          SizedBox(height: 56, child: ElevatedButton(
            onPressed: state.autoModeEnabled || state.isStarting ? null : (state.isLogging ? state.stopLogging : state.startLogging),
            style: ElevatedButton.styleFrom(backgroundColor: state.isLogging ? RHSColors.danger : RHSColors.success, disabledBackgroundColor: RHSColors.bgSecondary, foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
            child: state.isStarting ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
              : Row(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(state.autoModeEnabled ? Icons.lock : (state.isLogging ? Icons.stop : Icons.play_arrow)), const SizedBox(width: 8), Text(state.autoModeEnabled ? 'Auto Mode Active' : (state.isLogging ? 'Stop Monitoring' : 'Start Monitoring'), style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700))]),
          )),
          const SizedBox(height: 12),
          Text(state.autoModeEnabled ? (state.isLogging ? 'Road Health AI is Analyzing road conditions...' : 'Drive to auto-start monitoring') : 'Every drive improves our map', textAlign: TextAlign.center, style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.textSecondary, fontStyle: FontStyle.italic)),
          const SizedBox(height: 24),
          if (state.events.isNotEmpty) ...[
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [Text('Recent Detections', style: GoogleFonts.dmSans(fontWeight: FontWeight.w700, color: RHSColors.textPrimary)), Text('${state.events.length} total', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.textSecondary))]),
            const SizedBox(height: 12),
            ...state.events.reversed.take(5).map((e) => _eventTile(e)),
          ],
        ]),
      ),
    );
  }

  Widget _eventTile(DetectionEvent e) {
    final severityColor = RHSColors.severityColor(e.severityIndex);
    return Container(margin: const EdgeInsets.only(bottom: 8), padding: const EdgeInsets.all(14), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border(left: BorderSide(color: severityColor, width: 4))),
      child: Row(children: [
        Container(width: 42, height: 42, decoration: BoxDecoration(color: severityColor.withOpacity(0.12), borderRadius: BorderRadius.circular(10)), child: Icon(e.type.icon, color: severityColor, size: 22)),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(e.type.displayName, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600, fontSize: 14)),
          const SizedBox(height: 2),
          Text('${RHSColors.severityLabel(e.severityIndex)} • ${e.roadName ?? 'Unknown road'}', style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary)),
        ])),
        Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4), decoration: BoxDecoration(color: severityColor.withOpacity(0.12), borderRadius: BorderRadius.circular(6)), child: Text(e.severityIndex.toStringAsFixed(0), style: GoogleFonts.spaceMono(fontSize: 13, fontWeight: FontWeight.w700, color: severityColor))),
          if (e.photoPath != null && e.photoPath!.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 4), child: Icon(Icons.camera_alt, color: RHSColors.primary, size: 16)),
        ]),
      ]),
    );
  }

  Widget _buildStatCard(String label, String value, Color color) {
    return Container(padding: const EdgeInsets.all(16), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: RHSColors.textSecondary)),
        const SizedBox(height: 8),
        Text(value, style: GoogleFonts.spaceMono(fontSize: 24, fontWeight: FontWeight.w700, color: color)),
      ]),
    );
  }

  Widget _buildImpactStat(String value, String label) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(value, style: GoogleFonts.spaceMono(fontSize: 20, fontWeight: FontWeight.w700, color: Colors.white)),
      Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: Colors.white.withOpacity(0.8))),
    ]);
  }
}

// ============================================================================
// MAP SCREEN
// ============================================================================

class MapScreen extends StatefulWidget {
  final RoadHealthState appState;
  const MapScreen({super.key, required this.appState});
  @override State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  final MapController _mapController = MapController();
  DefectCluster? _selectedCluster;

  @override void initState() { super.initState(); widget.appState.addListener(_onStateChange); }
  @override void dispose() { widget.appState.removeListener(_onStateChange); super.dispose(); }
  void _onStateChange() { if (mounted) setState(() {}); }

  LatLng get _mapCenter {
    if (widget.appState.clusters.isNotEmpty) {
      final lats = widget.appState.clusters.map((c) => c.centroid.latitude);
      final lons = widget.appState.clusters.map((c) => c.centroid.longitude);
      return LatLng(lats.reduce((a, b) => a + b) / lats.length, lons.reduce((a, b) => a + b) / lons.length);
    }
    if (widget.appState.currentPosition != null) return LatLng(widget.appState.currentPosition!.latitude, widget.appState.currentPosition!.longitude);
    return const LatLng(-26.2041, 28.0473); // Johannesburg default
  }

  @override
  Widget build(BuildContext context) {
    final state = widget.appState;
    return Scaffold(
      body: Stack(children: [
        FlutterMap(
          mapController: _mapController,
          options: MapOptions(initialCenter: _mapCenter, initialZoom: 12, onTap: (_, __) => setState(() => _selectedCluster = null)),
          children: [
            TileLayer(urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', userAgentPackageName: 'com.roadhealth.app'),
            MarkerLayer(markers: state.clusters.map((cluster) {
              final color = RHSColors.severityColor(cluster.avgSeverity);
              final isSelected = _selectedCluster?.id == cluster.id;
              final size = 32.0 + (cluster.detectionCount.clamp(1, 10) * 3);
              return Marker(
                point: cluster.centroid, width: size + (isSelected ? 8 : 0), height: size + (isSelected ? 8 : 0),
                child: GestureDetector(onTap: () => setState(() => _selectedCluster = cluster),
                  child: AnimatedContainer(duration: const Duration(milliseconds: 200),
                    decoration: BoxDecoration(color: color.withOpacity(isSelected ? 1.0 : 0.85), shape: BoxShape.circle, border: Border.all(color: Colors.white, width: isSelected ? 3 : 2), boxShadow: [BoxShadow(color: color.withOpacity(0.4), blurRadius: isSelected ? 12 : 6, spreadRadius: isSelected ? 2 : 0)]),
                    child: Center(child: Text('${cluster.detectionCount}', style: GoogleFonts.spaceMono(fontSize: 12, fontWeight: FontWeight.w700, color: Colors.white))),
                  ),
                ),
              );
            }).toList()),
            if (state.currentPosition != null)
              MarkerLayer(markers: [Marker(point: LatLng(state.currentPosition!.latitude, state.currentPosition!.longitude), width: 24, height: 24,
                child: Container(decoration: BoxDecoration(color: RHSColors.primary, shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 3), boxShadow: [BoxShadow(color: RHSColors.primary.withOpacity(0.3), blurRadius: 8)])))]),
          ],
        ),
        // Top stats bar
        SafeArea(child: Padding(padding: const EdgeInsets.all(16), child: Row(children: [
          Expanded(child: Container(padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.08), blurRadius: 12, offset: const Offset(0, 4))]),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
              _buildMapStat('${state.clusters.length}', 'Clusters'),
              Container(width: 1, height: 30, color: RHSColors.bgSecondary),
              _buildMapStat('${state.events.length}', 'Detections'),
              Container(width: 1, height: 30, color: RHSColors.bgSecondary),
              _buildMapStat('${state.clusters.where((c) => c.avgSeverity >= 50).length}', 'High Sev'),
            ]),
          )),
          const SizedBox(width: 12),
          Container(decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.08), blurRadius: 12, offset: const Offset(0, 4))]),
            child: IconButton(onPressed: () => _mapController.move(_mapCenter, 12), icon: const Icon(Icons.my_location, color: RHSColors.primary))),
        ]))),
        // Legend
        Positioned(left: 16, bottom: 100, child: Container(padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.08), blurRadius: 12)]),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
            Text('Severity', style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600, color: RHSColors.textSecondary)),
            const SizedBox(height: 8),
            _legendItem(RHSColors.severityCritical, 'Critical (75+)'),
            _legendItem(RHSColors.severityHigh, 'High (50-74)'),
            _legendItem(RHSColors.severityMedium, 'Medium (25-49)'),
            _legendItem(RHSColors.severityLow, 'Low (<25)'),
          ]),
        )),
        // Selected cluster detail
        if (_selectedCluster != null) Positioned(left: 16, right: 16, bottom: 24, child: _buildClusterDetail(_selectedCluster!)),
        // Empty state
        if (state.clusters.isEmpty) Center(child: Container(margin: const EdgeInsets.all(32), padding: const EdgeInsets.all(24), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.08), blurRadius: 20)]),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.map_outlined, size: 48, color: RHSColors.textMuted),
            const SizedBox(height: 16),
            Text('No Detections Yet', style: GoogleFonts.dmSans(fontSize: 18, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
            const SizedBox(height: 8),
            Text('Start monitoring or enable demo mode for sample data.', textAlign: TextAlign.center, style: GoogleFonts.dmSans(fontSize: 14, color: RHSColors.textSecondary)),
            const SizedBox(height: 16),
            ElevatedButton.icon(onPressed: () => widget.appState.enableDemoMode(), icon: const Icon(Icons.science_outlined), label: const Text('Load Demo Data'), style: ElevatedButton.styleFrom(backgroundColor: RHSColors.primary, foregroundColor: Colors.white)),
          ]),
        )),
      ]),
    );
  }

  Widget _buildMapStat(String value, String label) => Column(mainAxisSize: MainAxisSize.min, children: [
    Text(value, style: GoogleFonts.spaceMono(fontSize: 18, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
    Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: RHSColors.textSecondary)),
  ]);

  Widget _legendItem(Color color, String label) => Padding(padding: const EdgeInsets.only(bottom: 6), child: Row(mainAxisSize: MainAxisSize.min, children: [
    Container(width: 12, height: 12, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
    const SizedBox(width: 8),
    Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: RHSColors.textSecondary)),
  ]));

  Widget _buildClusterDetail(DefectCluster cluster) {
    final color = RHSColors.severityColor(cluster.avgSeverity);
    return Container(padding: const EdgeInsets.all(16), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.12), blurRadius: 20, offset: const Offset(0, 8))]),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
        Row(children: [
          Container(width: 48, height: 48, decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(12)), child: Icon(cluster.dominantType.icon, color: color, size: 26)),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(cluster.roadName ?? 'Unknown Road', style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
            Text(cluster.ward ?? 'Unknown Ward', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.textSecondary)),
          ])),
          Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6), decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
            child: Text(RHSColors.severityLabel(cluster.avgSeverity), style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w700, color: color))),
        ]),
        const SizedBox(height: 16),
        Row(children: [
          _detailChip(Icons.warning_amber, '${cluster.detectionCount} detections'),
          const SizedBox(width: 12),
          _detailChip(Icons.speed, 'Sev: ${cluster.avgSeverity.toStringAsFixed(0)}'),
          const SizedBox(width: 12),
          _detailChip(Icons.category, cluster.dominantType.displayName),
        ]),
        if (cluster.firstSeen != null) ...[
          const SizedBox(height: 12),
          Text('First: ${_formatDate(cluster.firstSeen!)}  •  Last: ${_formatDate(cluster.lastSeen!)}', style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textMuted)),
        ],
      ]),
    );
  }

  Widget _detailChip(IconData icon, String label) => Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6), decoration: BoxDecoration(color: RHSColors.bgSecondary, borderRadius: BorderRadius.circular(8)),
    child: Row(mainAxisSize: MainAxisSize.min, children: [Icon(icon, size: 14, color: RHSColors.textSecondary), const SizedBox(width: 4), Text(label, style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary))]));

  String _formatDate(DateTime date) {
    final diff = DateTime.now().difference(date);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${date.day}/${date.month}';
  }
}

// ============================================================================
// WORKLIST SCREEN
// ============================================================================

class WorklistScreen extends StatefulWidget {
  final RoadHealthState appState;
  const WorklistScreen({super.key, required this.appState});
  @override State<WorklistScreen> createState() => _WorklistScreenState();
}

class _WorklistScreenState extends State<WorklistScreen> {
  @override void initState() { super.initState(); widget.appState.addListener(_onStateChange); }
  @override void dispose() { widget.appState.removeListener(_onStateChange); super.dispose(); }
  void _onStateChange() { if (mounted) setState(() {}); }

  @override
  Widget build(BuildContext context) {
    final state = widget.appState;
    final clusters = state.filteredClusters;
    final criticalCount = clusters.where((c) => c.avgSeverity >= 75).length;
    final highCount = clusters.where((c) => c.avgSeverity >= 50 && c.avgSeverity < 75).length;

    return Scaffold(
      body: SafeArea(child: Column(children: [
        Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(color: Colors.white, boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10, offset: const Offset(0, 4))]),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Maintenance Worklist', style: GoogleFonts.dmSans(fontSize: 24, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
            const SizedBox(height: 4),
            Text('Prioritized defect candidates for action', style: GoogleFonts.dmSans(fontSize: 14, color: RHSColors.textSecondary)),
            const SizedBox(height: 16),
            Row(children: [
              _buildQuickStat('${clusters.length}', 'Total', RHSColors.primary),
              const SizedBox(width: 12),
              _buildQuickStat('$criticalCount', 'Critical', RHSColors.severityCritical),
              const SizedBox(width: 12),
              _buildQuickStat('$highCount', 'High', RHSColors.severityHigh),
            ]),
            const SizedBox(height: 16),
            SingleChildScrollView(scrollDirection: Axis.horizontal, child: Row(children: [
              _buildFilterChip(label: 'Sort: ${_sortLabel(state.worklistSortBy)}', icon: Icons.sort, onTap: () => _showSortDialog()),
              const SizedBox(width: 8),
              _buildFilterChip(label: state.worklistTypeFilter?.displayName ?? 'All Types', icon: Icons.category, isActive: state.worklistTypeFilter != null, onTap: () => _showTypeFilterDialog()),
              const SizedBox(width: 8),
              _buildFilterChip(label: state.worklistWardFilter ?? 'All Wards', icon: Icons.location_city, isActive: state.worklistWardFilter != null, onTap: () => _showWardFilterDialog()),
              if (state.worklistTypeFilter != null || state.worklistWardFilter != null) ...[
                const SizedBox(width: 8),
                GestureDetector(onTap: () => setState(() { state.worklistTypeFilter = null; state.worklistWardFilter = null; }),
                  child: Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8), decoration: BoxDecoration(color: RHSColors.danger.withOpacity(0.1), borderRadius: BorderRadius.circular(8)),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [Icon(Icons.clear, size: 16, color: RHSColors.danger), const SizedBox(width: 4), Text('Clear', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.danger, fontWeight: FontWeight.w500))]))),
              ],
            ])),
          ]),
        ),
        Expanded(child: clusters.isEmpty ? _buildEmptyState() : ListView.builder(
          padding: const EdgeInsets.all(16), itemCount: clusters.length,
          itemBuilder: (context, index) => _buildWorklistItem(clusters[index], index + 1),
        )),
      ])),
      floatingActionButton: clusters.isNotEmpty ? FloatingActionButton.extended(
        onPressed: () => widget.appState.exportSessionFiles(),
        backgroundColor: RHSColors.primary,
        icon: const Icon(Icons.upload_file, color: Colors.white),
        label: Text('Export', style: GoogleFonts.dmSans(color: Colors.white, fontWeight: FontWeight.w600)),
      ) : null,
    );
  }

  Widget _buildQuickStat(String value, String label, Color color) => Expanded(child: Container(padding: const EdgeInsets.symmetric(vertical: 12), decoration: BoxDecoration(color: color.withOpacity(0.08), borderRadius: BorderRadius.circular(10)),
    child: Column(children: [Text(value, style: GoogleFonts.spaceMono(fontSize: 22, fontWeight: FontWeight.w700, color: color)), Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: color))])));

  Widget _buildFilterChip({required String label, required IconData icon, required VoidCallback onTap, bool isActive = false}) {
    return GestureDetector(onTap: onTap, child: Container(padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8), decoration: BoxDecoration(color: isActive ? RHSColors.primary.withOpacity(0.1) : RHSColors.bgSecondary, borderRadius: BorderRadius.circular(8), border: isActive ? Border.all(color: RHSColors.primary, width: 1) : null),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 16, color: isActive ? RHSColors.primary : RHSColors.textSecondary),
        const SizedBox(width: 6),
        Text(label, style: GoogleFonts.dmSans(fontSize: 13, color: isActive ? RHSColors.primary : RHSColors.textSecondary, fontWeight: isActive ? FontWeight.w600 : FontWeight.w400)),
        const SizedBox(width: 4),
        Icon(Icons.arrow_drop_down, size: 18, color: isActive ? RHSColors.primary : RHSColors.textMuted),
      ])));
  }

  Widget _buildWorklistItem(DefectCluster cluster, int rank) {
    final color = RHSColors.severityColor(cluster.avgSeverity);
    return Container(margin: const EdgeInsets.only(bottom: 12), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: RHSColors.bgSecondary)),
      child: Column(children: [
        Padding(padding: const EdgeInsets.all(16), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Container(width: 36, height: 36, decoration: BoxDecoration(gradient: rank <= 3 ? RHSColors.dangerGradient : null, color: rank > 3 ? RHSColors.bgSecondary : null, borderRadius: BorderRadius.circular(10)),
            child: Center(child: Text('#$rank', style: GoogleFonts.spaceMono(fontSize: 13, fontWeight: FontWeight.w700, color: rank <= 3 ? Colors.white : RHSColors.textSecondary)))),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [Icon(cluster.dominantType.icon, size: 18, color: color), const SizedBox(width: 6), Text(cluster.dominantType.displayName, style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w700, color: RHSColors.textPrimary))]),
            const SizedBox(height: 4),
            Text(cluster.roadName ?? 'Unknown Road', style: GoogleFonts.dmSans(fontSize: 14, color: RHSColors.textPrimary)),
            const SizedBox(height: 2),
            Text(cluster.ward ?? 'Unknown Ward', style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary)),
          ])),
          Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
            Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6), decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
              child: Text(RHSColors.severityLabel(cluster.avgSeverity), style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700, color: color))),
            const SizedBox(height: 6),
            Text('Score: ${cluster.priorityScore.toStringAsFixed(0)}', style: GoogleFonts.spaceMono(fontSize: 11, color: RHSColors.textMuted)),
          ]),
        ])),
        Container(padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10), decoration: BoxDecoration(color: RHSColors.bgPrimary, borderRadius: const BorderRadius.only(bottomLeft: Radius.circular(14), bottomRight: Radius.circular(14))),
          child: Row(children: [
            _worklistStat(Icons.repeat, '${cluster.detectionCount} detections'),
            const SizedBox(width: 16),
            _worklistStat(Icons.speed, 'Sev: ${cluster.avgSeverity.toStringAsFixed(0)}'),
            const Spacer(),
            if (cluster.lastSeen != null) Text('Last: ${_formatDate(cluster.lastSeen!)}', style: GoogleFonts.dmSans(fontSize: 11, color: RHSColors.textMuted)),
          ])),
      ]),
    );
  }

  Widget _worklistStat(IconData icon, String label) => Row(mainAxisSize: MainAxisSize.min, children: [Icon(icon, size: 14, color: RHSColors.textMuted), const SizedBox(width: 4), Text(label, style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary))]);

  Widget _buildEmptyState() => Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
    Icon(Icons.assignment_outlined, size: 64, color: RHSColors.textMuted),
    const SizedBox(height: 16),
    Text('No Defects in Worklist', style: GoogleFonts.dmSans(fontSize: 18, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
    const SizedBox(height: 8),
    Text('Start monitoring or enable demo mode.', textAlign: TextAlign.center, style: GoogleFonts.dmSans(fontSize: 14, color: RHSColors.textSecondary)),
    const SizedBox(height: 24),
    ElevatedButton.icon(onPressed: () => widget.appState.enableDemoMode(), icon: const Icon(Icons.science_outlined), label: const Text('Load Demo Data'), style: ElevatedButton.styleFrom(backgroundColor: RHSColors.primary, foregroundColor: Colors.white)),
  ]));

  String _sortLabel(String sortBy) { switch (sortBy) { case 'severity': return 'Severity'; case 'date': return 'Recent'; default: return 'Priority'; } }

  void _showSortDialog() {
    showModalBottomSheet(context: context, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (context) => Container(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Sort By', style: GoogleFonts.dmSans(fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 16),
        _sortOption('priority', 'Priority Score', 'Combines severity, frequency, and confidence'),
        _sortOption('severity', 'Severity', 'Highest severity first'),
        _sortOption('date', 'Most Recent', 'Most recently detected first'),
      ])));
  }

  Widget _sortOption(String value, String title, String subtitle) {
    final isSelected = widget.appState.worklistSortBy == value;
    return ListTile(onTap: () { setState(() => widget.appState.worklistSortBy = value); Navigator.pop(context); },
      leading: Icon(isSelected ? Icons.radio_button_checked : Icons.radio_button_off, color: isSelected ? RHSColors.primary : RHSColors.textMuted),
      title: Text(title, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)), subtitle: Text(subtitle, style: GoogleFonts.dmSans(fontSize: 12)));
  }

  void _showTypeFilterDialog() {
    showModalBottomSheet(context: context, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (context) => Container(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Filter by Type', style: GoogleFonts.dmSans(fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 16),
        _typeOption(null, 'All Types'),
        ...DefectType.values.map((t) => _typeOption(t, t.displayName)),
      ])));
  }

  Widget _typeOption(DefectType? type, String label) {
    final isSelected = widget.appState.worklistTypeFilter == type;
    return ListTile(onTap: () { setState(() => widget.appState.worklistTypeFilter = type); Navigator.pop(context); },
      leading: Icon(type?.icon ?? Icons.category, color: isSelected ? RHSColors.primary : RHSColors.textMuted),
      title: Text(label, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)), trailing: isSelected ? const Icon(Icons.check, color: RHSColors.primary) : null);
  }

  void _showWardFilterDialog() {
    final wards = widget.appState.availableWards.toList()..sort();
    showModalBottomSheet(context: context, shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (context) => Container(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Filter by Ward', style: GoogleFonts.dmSans(fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 16),
        _wardOption(null, 'All Wards'),
        ...wards.map((w) => _wardOption(w, w)),
      ])));
  }

  Widget _wardOption(String? ward, String label) {
    final isSelected = widget.appState.worklistWardFilter == ward;
    return ListTile(onTap: () { setState(() => widget.appState.worklistWardFilter = ward); Navigator.pop(context); },
      leading: Icon(Icons.location_city, color: isSelected ? RHSColors.primary : RHSColors.textMuted),
      title: Text(label, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)), trailing: isSelected ? const Icon(Icons.check, color: RHSColors.primary) : null);
  }

  String _formatDate(DateTime date) { final diff = DateTime.now().difference(date); if (diff.inDays == 0) return 'Today'; if (diff.inDays == 1) return 'Yesterday'; if (diff.inDays < 7) return '${diff.inDays}d ago'; return '${date.day}/${date.month}'; }
}

// ============================================================================
// SETTINGS SCREEN
// ============================================================================

class SettingsScreen extends StatefulWidget {
  final RoadHealthState appState;
  const SettingsScreen({super.key, required this.appState});
  @override State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  @override
  Widget build(BuildContext context) {
    final state = widget.appState;
    return Scaffold(
      body: SafeArea(child: ListView(padding: const EdgeInsets.all(20), children: [
        Text('Settings', style: GoogleFonts.dmSans(fontSize: 28, fontWeight: FontWeight.w700, color: RHSColors.textPrimary)),
        const SizedBox(height: 8),
        Text('Configure monitoring and export options', style: GoogleFonts.dmSans(fontSize: 14, color: RHSColors.textSecondary)),
        const SizedBox(height: 24),
        Container(padding: const EdgeInsets.all(16), decoration: BoxDecoration(gradient: state.demoModeEnabled ? RHSColors.primaryGradient : null, color: state.demoModeEnabled ? null : Colors.white, borderRadius: BorderRadius.circular(14), border: state.demoModeEnabled ? null : Border.all(color: RHSColors.bgSecondary)),
          child: Row(children: [
            Container(width: 44, height: 44, decoration: BoxDecoration(color: state.demoModeEnabled ? Colors.white.withOpacity(0.2) : RHSColors.bgSecondary, borderRadius: BorderRadius.circular(12)),
              child: Icon(Icons.science_outlined, color: state.demoModeEnabled ? Colors.white : RHSColors.textMuted)),
            const SizedBox(width: 14),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Demo Mode', style: GoogleFonts.dmSans(fontWeight: FontWeight.w700, color: state.demoModeEnabled ? Colors.white : RHSColors.textPrimary)),
              Text(state.demoModeEnabled ? 'Using sample Johannesburg data' : 'Load sample data for presentations', style: GoogleFonts.dmSans(fontSize: 12, color: state.demoModeEnabled ? Colors.white.withOpacity(0.8) : RHSColors.textSecondary)),
            ])),
            Switch(value: state.demoModeEnabled, onChanged: (val) { if (val) state.enableDemoMode(); else state.disableDemoMode(); setState(() {}); }, activeColor: Colors.white, activeTrackColor: Colors.white.withOpacity(0.3)),
          ])),
        const SizedBox(height: 24),
        _sectionHeader('Feedback'),
        _buildSettingSwitch(icon: Icons.volume_up, title: 'Audio Alerts', subtitle: 'Play sound on detection', value: state.audioEnabled, onChanged: (val) => setState(() => state.audioEnabled = val)),
        _buildSettingSwitch(icon: Icons.vibration, title: 'Haptic Feedback', subtitle: 'Vibrate on detection', value: state.hapticEnabled, onChanged: (val) => setState(() => state.hapticEnabled = val)),
        const SizedBox(height: 24),
        _sectionHeader('Evidence Capture'),
        _buildSettingSwitch(icon: Icons.camera_alt, title: 'Camera Enabled', subtitle: 'Allow camera usage', value: state.cameraEnabled, onChanged: (val) => setState(() => state.cameraEnabled = val)),
        _buildSettingSwitch(icon: Icons.photo_library, title: 'Capture Evidence', subtitle: 'Save photos for verification', value: state.evidenceCaptureEnabled, onChanged: (val) => setState(() => state.evidenceCaptureEnabled = val)),
        const SizedBox(height: 24),
        _sectionHeader('Data Export'),
        _buildActionTile(icon: Icons.upload_file, title: 'Export Session Files', subtitle: 'Shares CSV, GeoJSON for GIS', onTap: () async { await state.exportSessionFiles(); if (mounted) setState(() {}); }),
        const SizedBox(height: 24),
        _sectionHeader('About'),
        Container(padding: const EdgeInsets.all(16), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: RHSColors.bgSecondary)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.asset('assets/images/rhs_logo.png', width: 56, height: 56, fit: BoxFit.contain)),
              const SizedBox(width: 14),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('Road Health Score', style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700)), Text('Version 2.0 Pilot', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.textSecondary))])),
            ]),
            const SizedBox(height: 16),
            Text('Municipal-grade road condition intelligence via crowdsourced sensing and AI verification.', style: GoogleFonts.dmSans(fontSize: 13, color: RHSColors.textSecondary, height: 1.5)),
          ])),
      ])),
    );
  }

  Widget _sectionHeader(String title) => Padding(padding: const EdgeInsets.only(bottom: 12), child: Text(title.toUpperCase(), style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w700, color: RHSColors.textMuted, letterSpacing: 1)));

  Widget _buildSettingSwitch({required IconData icon, required String title, required String subtitle, required bool value, required ValueChanged<bool> onChanged}) {
    return Container(margin: const EdgeInsets.only(bottom: 8), padding: const EdgeInsets.all(14), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: RHSColors.bgSecondary)),
      child: Row(children: [
        Container(width: 40, height: 40, decoration: BoxDecoration(color: value ? RHSColors.primary.withOpacity(0.1) : RHSColors.bgSecondary, borderRadius: BorderRadius.circular(10)), child: Icon(icon, size: 20, color: value ? RHSColors.primary : RHSColors.textMuted)),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)), Text(subtitle, style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary))])),
        Switch(value: value, onChanged: onChanged, activeColor: RHSColors.primary),
      ]));
  }

  Widget _buildActionTile({required IconData icon, required String title, required String subtitle, required VoidCallback onTap}) {
    return GestureDetector(onTap: onTap, child: Container(padding: const EdgeInsets.all(14), decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: RHSColors.bgSecondary)),
      child: Row(children: [
        Container(width: 40, height: 40, decoration: BoxDecoration(color: RHSColors.primary.withOpacity(0.1), borderRadius: BorderRadius.circular(10)), child: Icon(icon, size: 20, color: RHSColors.primary)),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: GoogleFonts.dmSans(fontWeight: FontWeight.w600)), Text(subtitle, style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary))])),
        const Icon(Icons.chevron_right, color: RHSColors.textMuted),
      ])));
  }
}
