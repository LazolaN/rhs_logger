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

class RHSColors {
  static const Color primary = Color(0xFF0D7377);
  static const Color primaryLight = Color(0xFF14A3A8);
  static const Color success = Color(0xFF2ECC71);
  static const Color warning = Color(0xFFF39C12);
  static const Color danger = Color(0xFFE74C3C);
  static const Color bgPrimary = Color(0xFFF7F9FC);
  static const Color bgSecondary = Color(0xFFEEF2F7);
  static const Color bgCard = Color(0xFFFFFFFF);
  static const Color textPrimary = Color(0xFF2D3436);
  static const Color textSecondary = Color(0xFF636E72);
  static const Color textMuted = Color(0xFFB2BEC3);

  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [primary, primaryLight],
  );
}

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
  ));

  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Camera init error: $e');
  }

  runApp(const RoadHealthApp());
}

class RoadHealthApp extends StatelessWidget {
  const RoadHealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Health',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: RHSColors.primary,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: RHSColors.bgPrimary,
        textTheme: GoogleFonts.dmSansTextTheme(),
        cardTheme: const CardThemeData(
          elevation: 0,
          color: RHSColors.bgCard,
        ),
      ),
      home: const MainNavigationPage(),
    );
  }
}

class MainNavigationPage extends StatefulWidget {
  const MainNavigationPage({super.key});

  @override
  State<MainNavigationPage> createState() => _MainNavigationPageState();
}

class _MainNavigationPageState extends State<MainNavigationPage> {
  int _currentIndex = 0;
  final RoadHealthState _appState = RoadHealthState();

  @override
  void initState() {
    super.initState();
    _appState.initialize();
  }

  @override
  void dispose() {
    _appState.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          MonitorScreen(appState: _appState),
          MapScreen(appState: _appState),
          SettingsScreen(appState: _appState),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 20,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildNavItem(0, Icons.sensors, Icons.sensors_outlined, 'Monitor'),
              _buildNavItem(1, Icons.map, Icons.map_outlined, 'Map'),
              _buildNavItem(2, Icons.settings, Icons.settings_outlined, 'Settings'),
            ],
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
        decoration: BoxDecoration(
          color: isActive ? RHSColors.primary.withOpacity(0.1) : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isActive ? activeIcon : inactiveIcon,
              color: isActive ? RHSColors.primary : RHSColors.textMuted,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 11,
                fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                color: isActive ? RHSColors.primary : RHSColors.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/* -----------------------------
   V2 Domain Entities
--------------------------------*/

enum DefectType { pothole, speedBump, roughRoad, unknown }

class DetectionEvent {
  DetectionEvent({
    required this.id,
    required this.timestamp,
    required this.latitude,
    required this.longitude,
    required this.speedKmh,
    required this.severityIndex,
    required this.rawPeak,
    required this.rawP2P,
    required this.modelConfidence,
    required this.type,
    this.photoPath,
    this.visionLabel,
    this.visionConfidence,
  });

  final String id;
  final DateTime timestamp;
  final double latitude;
  final double longitude;
  final double speedKmh;

  // Severity index is an impact-based prioritization signal, not depth.
  final double severityIndex;

  // Raw features from the window
  final double rawPeak;
  final double rawP2P;

  // Detection engine confidence (inertial)
  final double modelConfidence;

  // Classified type (may be refined by vision)
  final DefectType type;

  // Evidence (optional)
  String? photoPath;
  String? visionLabel;
  double? visionConfidence;

  Map<String, dynamic> toJson() => {
        'id': id,
        'timestamp': timestamp.toIso8601String(),
        'latitude': latitude,
        'longitude': longitude,
        'speed_kmh': speedKmh,
        'severity_index': severityIndex,
        'raw_peak': rawPeak,
        'raw_p2p': rawP2P,
        'inertial_confidence': modelConfidence,
        'type': type.name,
        'photo': photoPath ?? '',
        'vision_label': visionLabel ?? '',
        'vision_confidence': visionConfidence ?? '',
      };
}

/* -----------------------------
   V2 Detection Engine
--------------------------------*/

class _AccelSample {
  _AccelSample(this.t, this.ax, this.ay, this.az);
  final DateTime t;
  final double ax;
  final double ay;
  final double az;

  double get magnitude => sqrt(ax * ax + ay * ay + az * az);
}

class DetectionEngine {
  // Tuning parameters (start conservative, then calibrate using pilot data)
  static const int windowMs = 450;
  static const int minWindowSamples = 8;
  static const Duration cooldown = Duration(milliseconds: 1500);
  static const double minSpeedKmh = 5.0;
  static const double maxReasonableSpeedKmh = 160.0;

  // Dynamic threshold model:
  // Higher speed produces higher impact energy for benign features too, so threshold increases mildly with speed.
  // Calibrate this per vehicle type in V3 (fleet profiles).
  static double thresholdForSpeed(double speedKmh) {
    final s = speedKmh.clamp(0.0, maxReasonableSpeedKmh);
    return 14.0 + 0.10 * s; // example: 14 at 0kmh, 24 at 100kmh
  }

  static double severityIndex(double peakMag, double p2p, double speedKmh) {
    // A simple, stable index for prioritization:
    // speed-normalized impact + shape term.
    final denom = max(8.0, speedKmh);
    final base = (peakMag / denom) * 100.0;
    final shape = (p2p / denom) * 40.0;
    return (base + shape).clamp(0.0, 100.0);
  }

  final List<_AccelSample> _buf = [];
  DateTime? _lastTrigger;

  // Simple state
  bool _armed = false;
  DateTime? _armStart;

  DetectionEvent? ingest({
    required _AccelSample sample,
    required Position? position,
    required double speedKmh,
  }) {
    // Guardrails
    if (position == null) return null;
    if (speedKmh < minSpeedKmh) return null;
    if (_lastTrigger != null && sample.t.difference(_lastTrigger!) < cooldown) return null;

    _buf.add(sample);

    // Keep buffer bounded by windowMs
    final cutoff = sample.t.subtract(const Duration(milliseconds: windowMs));
    while (_buf.isNotEmpty && _buf.first.t.isBefore(cutoff)) {
      _buf.removeAt(0);
    }

    if (_buf.length < minWindowSamples) return null;

    // Compute window stats
    double peak = 0.0;
    double minMag = double.infinity;
    double maxMag = 0.0;

    for (final s in _buf) {
      final m = s.magnitude;
      peak = max(peak, m);
      minMag = min(minMag, m);
      maxMag = max(maxMag, m);
    }
    final p2p = (maxMag - minMag).abs();

    final thr = thresholdForSpeed(speedKmh);

    // Arming logic reduces single-sample spikes
    if (!_armed && (peak > thr)) {
      _armed = true;
      _armStart = sample.t;
      return null;
    }

    if (_armed) {
      // Confirm trigger when we have a stable peak in the armed window
      final armedAge = sample.t.difference(_armStart!);
      if (armedAge.inMilliseconds >= 180) {
        // Decide if this is a candidate event
        final confidence = _inertialConfidence(peak, p2p, thr);

        if (confidence >= 0.75) {
          _armed = false;
          _armStart = null;
          _lastTrigger = sample.t;

          final sev = severityIndex(peak, p2p, speedKmh);

          // Initial coarse classification from inertial only.
          // Vision verifier may refine later.
          final guess = _coarseType(peak, p2p, speedKmh);

          return DetectionEvent(
            id: _makeId(sample.t, position.latitude, position.longitude),
            timestamp: sample.t,
            latitude: position.latitude,
            longitude: position.longitude,
            speedKmh: speedKmh,
            severityIndex: sev,
            rawPeak: peak,
            rawP2P: p2p,
            modelConfidence: confidence,
            type: guess,
          );
        } else {
          // Disarm if confidence is too low after a short armed period
          if (armedAge.inMilliseconds >= 350) {
            _armed = false;
            _armStart = null;
          }
        }
      }
    }

    return null;
  }

  double _inertialConfidence(double peak, double p2p, double thr) {
    // Confidence based on how far above threshold and how pronounced the shape is.
    final above = (peak - thr).clamp(0.0, 20.0) / 20.0;
    final shape = (p2p.clamp(0.0, 10.0)) / 10.0;
    final c = 0.65 * above + 0.35 * shape;
    return c.clamp(0.0, 1.0);
  }

  DefectType _coarseType(double peak, double p2p, double speedKmh) {
    // Heuristic: speed bumps often show strong, smoother p2p at moderate speeds,
    // potholes often show sharp peak and higher p2p at higher speed.
    if (speedKmh >= 12 && speedKmh <= 40 && p2p >= 5.5 && peak >= 20) {
      return DefectType.speedBump;
    }
    if (peak >= 22 && p2p >= 6.0) return DefectType.pothole;
    if (peak >= 18) return DefectType.roughRoad;
    return DefectType.unknown;
  }

  String _makeId(DateTime t, double lat, double lon) {
    final ts = t.millisecondsSinceEpoch;
    final h = (lat * 100000).round() ^ (lon * 100000).round() ^ ts;
    return '$ts-$h';
  }

  void reset() {
    _buf.clear();
    _armed = false;
    _armStart = null;
    _lastTrigger = null;
  }
}

/* -----------------------------
   V2 Vision Verifier Abstraction
--------------------------------*/

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

// Fallback verifier using ML Kit image labeling.
// For V2 pilot credibility, replace this with a custom on-device model (TFLite).
class MlKitVisionVerifier implements VisionVerifier {
  MlKitVisionVerifier({double confidenceThreshold = 0.55})
      : _labeler = ImageLabeler(options: ImageLabelerOptions(confidenceThreshold: confidenceThreshold));

  final ImageLabeler _labeler;

  @override
  Future<VisionResult?> verify(String imagePath) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);
      final labels = await _labeler.processImage(inputImage);

      if (labels.isEmpty) return null;

      // Keep the strongest label for logging, but only do a gentle refinement.
      labels.sort((a, b) => b.confidence.compareTo(a.confidence));
      final top = labels.first;
      final txt = top.label.toLowerCase();

      // Conservative mapping (avoid overclaiming)
      DefectType refined = DefectType.unknown;
      if (txt.contains('speed') || txt.contains('bump')) refined = DefectType.speedBump;
      if (txt.contains('road') || txt.contains('asphalt')) refined = DefectType.roughRoad;

      return VisionResult(top.label, top.confidence, refined);
    } catch (e) {
      debugPrint('Vision verify error: $e');
      return null;
    }
  }

  @override
  void close() {
    _labeler.close();
  }
}

/* -----------------------------
   RoadHealthState (V2)
--------------------------------*/

class RoadHealthState extends ChangeNotifier {
  // Session state
  bool isLogging = false;
  bool autoModeEnabled = false;
  bool isStarting = false;

  // UI telemetry
  double currentImpact = 0.0; // magnitude
  double maxImpact = 0.0;

  // Stats
  int totalSamples = 0;
  int detections = 0;
  int photosCaptured = 0;
  DateTime? sessionStartTime;
  Duration sessionDuration = Duration.zero;

  // Lifetime
  int lifetimeDetections = 0;
  double lifetimeKmMapped = 0.0;

  // GPS
  Position? currentPosition;
  double currentSpeedKmh = 0.0;
  DateTime? lastMovementTime;

  // Settings
  bool audioEnabled = true;
  bool hapticEnabled = true;
  bool cameraEnabled = true;
  bool evidenceCaptureEnabled = true; // V2: allow user to disable evidence capture separately

  // Timers/subscriptions
  StreamSubscription<UserAccelerometerEvent>? _accelSub;
  Timer? _gpsTimer;
  Timer? _flushTimer;
  Timer? _autoStopTimer;
  Timer? _sessionTimer;
  DateTime? _lastUiUpdate;

  // Engine
  final DetectionEngine _engine = DetectionEngine();

  // Evidence
  CameraController? cameraController;
  bool _isCapturing = false;
  final AudioPlayer _audioPlayer = AudioPlayer();
  VisionVerifier? _vision;

  // Data stores
  final List<Map<String, dynamic>> _telemetryBatch = []; // lightweight periodic telemetry
  final List<DetectionEvent> _events = [];

  // Exposed for UI
  List<DetectionEvent> get events => List.unmodifiable(_events);

  // Callbacks
  Function(String)? onError;
  Function(String)? onSuccess;

  // Constants
  static const double autoStartSpeedThreshold = 10.0;
  static const Duration autoStopIdleDuration = Duration(minutes: 3);
  static const Duration uiUpdateInterval = Duration(milliseconds: 150);

  Future<void> initialize() async {
    await _loadLifetimeStats();
    await _initializeCamera();
    _vision = MlKitVisionVerifier();
    _initForegroundTask();
    notifyListeners();
  }

  Future<void> _loadLifetimeStats() async {
    final prefs = await SharedPreferences.getInstance();
    lifetimeDetections = prefs.getInt('lifetime_detections') ?? 0;
    lifetimeKmMapped = prefs.getDouble('lifetime_km') ?? 0.0;
  }

  Future<void> _saveLifetimeStats() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('lifetime_detections', lifetimeDetections);
    await prefs.setDouble('lifetime_km', lifetimeKmMapped);
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      cameraEnabled = false;
      return;
    }
    try {
      cameraController = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await cameraController!.initialize();
    } catch (e) {
      debugPrint('Camera init error: $e');
      cameraEnabled = false;
    }
  }

  void _initForegroundTask() {
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'rhs_channel',
        channelName: 'Road Health Monitoring',
        channelDescription: 'Active road condition monitoring',
        channelImportance: NotificationChannelImportance.LOW,
        priority: NotificationPriority.LOW,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: true,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        eventAction: ForegroundTaskEventAction.nothing(),
        autoRunOnBoot: false,
        allowWakeLock: true,
        allowWifiLock: true,
      ),
    );
  }

  Future<bool> requestPermissions() async {
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        onError?.call('Location permission required to monitor roads');
        return false;
      }
    }
    if (permission == LocationPermission.deniedForever) {
      onError?.call('Location permission denied forever. Enable it in Settings.');
      return false;
    }

    if (Platform.isAndroid) {
      final notificationStatus = await Permission.notification.request();
      if (!notificationStatus.isGranted) {
        onError?.call('Notification permission required for background monitoring');
        return false;
      }
    }

    if (cameraEnabled && evidenceCaptureEnabled) {
      final cameraStatus = await Permission.camera.request();
      if (!cameraStatus.isGranted) {
        onError?.call('Camera permission denied. Evidence capture disabled.');
        evidenceCaptureEnabled = false;
      }
    }

    return true;
  }

  Future<void> startLogging() async {
    if (isLogging || isStarting) return;
    isStarting = true;
    notifyListeners();

    try {
      final ok = await requestPermissions();
      if (!ok) return;

      if (await FlutterForegroundTask.isRunningService) {
        await FlutterForegroundTask.restartService();
      } else {
        await FlutterForegroundTask.startService(
          notificationTitle: 'Road Health Active',
          notificationText: 'Monitoring road conditions...',
        );
      }

      isLogging = true;
      totalSamples = 0;
      detections = 0;
      photosCaptured = 0;
      maxImpact = 0.0;
      sessionStartTime = DateTime.now();
      sessionDuration = Duration.zero;
      _telemetryBatch.clear();
      _events.clear();
      _engine.reset();

      // Start streams
      _accelSub = userAccelerometerEventStream(
        samplingPeriod: const Duration(milliseconds: 20),
      ).listen(_onAccelEvent);

      _gpsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _updateGPS());
      _flushTimer = Timer.periodic(const Duration(seconds: 20), (_) => _flushToDisk());

      _sessionTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (sessionStartTime != null) {
          sessionDuration = DateTime.now().difference(sessionStartTime!);
          notifyListeners();
        }
      });

      if (autoModeEnabled) {
        lastMovementTime = DateTime.now();
        _autoStopTimer = Timer.periodic(const Duration(seconds: 10), (_) => _checkAutoStop());
      }

      onSuccess?.call('Monitoring started');
    } catch (e) {
      onError?.call('Failed to start: $e');
    } finally {
      isStarting = false;
      notifyListeners();
    }
  }

  Future<void> stopLogging() async {
    if (!isLogging) return;

    await _accelSub?.cancel();
    _accelSub = null;

    _gpsTimer?.cancel();
    _gpsTimer = null;

    _flushTimer?.cancel();
    _flushTimer = null;

    _autoStopTimer?.cancel();
    _autoStopTimer = null;

    _sessionTimer?.cancel();
    _sessionTimer = null;

    await _flushToDisk();
    await FlutterForegroundTask.stopService();

    lifetimeDetections += detections;
    await _saveLifetimeStats();

    isLogging = false;
    notifyListeners();

    onSuccess?.call('Session complete: $detections detections, $photosCaptured photos');
  }

  void toggleAutoMode() {
    autoModeEnabled = !autoModeEnabled;
    notifyListeners();

    if (autoModeEnabled && !isLogging) {
      _gpsTimer = Timer.periodic(const Duration(seconds: 2), (_) => _updateGPS());
      onSuccess?.call('Auto mode ON: starts at 10 km/h');
    } else if (!autoModeEnabled && !isLogging) {
      _gpsTimer?.cancel();
      _gpsTimer = null;
      onSuccess?.call('Manual mode enabled');
    }
  }

  void _onAccelEvent(UserAccelerometerEvent e) {
    final now = DateTime.now();

    // Compute mount-agnostic magnitude
    final sample = _AccelSample(now, e.x, e.y, e.z);
    final mag = sample.magnitude;
    currentImpact = mag;
    if (mag > maxImpact) maxImpact = mag;

    // Throttle UI updates
    if (_lastUiUpdate == null || now.difference(_lastUiUpdate!) >= uiUpdateInterval) {
      _lastUiUpdate = now;
      notifyListeners();
    }

    // Lightweight telemetry at ~10Hz (not every 50Hz)
    if (currentPosition != null && totalSamples % 5 == 0) {
      _telemetryBatch.add({
        'timestamp': now.toIso8601String(),
        'latitude': currentPosition!.latitude,
        'longitude': currentPosition!.longitude,
        'speed_kmh': currentSpeedKmh,
        'impact_mag': mag,
      });
    }

    totalSamples++;

    // Detection engine ingestion
    final event = _engine.ingest(
      sample: sample,
      position: currentPosition,
      speedKmh: currentSpeedKmh,
    );

    if (event != null) {
      _handleDetection(event);
    }
  }

  Future<void> _handleDetection(DetectionEvent event) async {
    detections++;

    if (hapticEnabled) {
      HapticFeedback.heavyImpact();
    }

    if (audioEnabled) {
      try {
        await _audioPlayer.play(AssetSource('sounds/beep.mp3'));
      } catch (_) {
        if (hapticEnabled) HapticFeedback.mediumImpact();
      }
    }

    _events.add(event);
    notifyListeners();

    // Evidence capture is optional and gated
    if (cameraEnabled &&
        evidenceCaptureEnabled &&
        cameraController != null &&
        cameraController!.value.isInitialized) {
      await _captureEvidenceFor(event);
    }
  }

  Future<void> _captureEvidenceFor(DetectionEvent event) async {
    if (_isCapturing) return;
    _isCapturing = true;

    try {
      final dir = await getApplicationDocumentsDirectory();
      final ts = DateTime.now().millisecondsSinceEpoch;
      final imagePath = '${dir.path}/rhs_event_${event.id}_$ts.jpg';

      final pic = await cameraController!.takePicture();
      await File(pic.path).copy(imagePath);

      event.photoPath = imagePath;
      photosCaptured++;

      // Run vision verifier (fallback)
      if (_vision != null) {
        final r = await _vision!.verify(imagePath);
        if (r != null) {
          event.visionLabel = r.label;
          event.visionConfidence = r.confidence;

          // Only accept refined type when confidence is high
          if (r.confidence >= 0.75 && r.refinedType != DefectType.unknown) {
            // keep event.type stable if it was pothole already
            // but allow speed bump refinement when strong
            if (event.type != DefectType.pothole || r.refinedType == DefectType.speedBump) {
              // ignore: prefer pothole if inertial says pothole
            }
          }
        }
      }

      notifyListeners();
    } catch (e) {
      debugPrint('Evidence capture error: $e');
    } finally {
      _isCapturing = false;
    }
  }

  Future<void> _updateGPS() async {
    try {
      final pos = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);

      final newSpeedKmh = (pos.speed * 3.6).isFinite ? pos.speed * 3.6 : 0.0;

      // Correct distance tracking:
      // compute distance from currentPosition -> new pos, then shift.
      if (isLogging && currentPosition != null) {
        final d = Geolocator.distanceBetween(
          currentPosition!.latitude,
          currentPosition!.longitude,
          pos.latitude,
          pos.longitude,
        );
        if (d > 5 && d < 1500) {
          lifetimeKmMapped += d / 1000.0;
        }
      }

      currentPosition = pos;
      currentSpeedKmh = newSpeedKmh;

      // Auto start
      if (autoModeEnabled && !isLogging && !isStarting && currentSpeedKmh > autoStartSpeedThreshold) {
        final ok = await requestPermissions();
        if (ok) await startLogging();
      }

      if (isLogging && currentSpeedKmh > 1.0) {
        lastMovementTime = DateTime.now();
      }
    } catch (e) {
      debugPrint('GPS error: $e');
    }
  }

  void _checkAutoStop() {
    if (!isLogging || !autoModeEnabled || lastMovementTime == null) return;
    if (DateTime.now().difference(lastMovementTime!) >= autoStopIdleDuration) {
      stopLogging();
    }
  }

  Future<void> _flushToDisk() async {
    try {
      // Write telemetry and events into separate files for usability.
      final dir = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];

      // Telemetry CSV
      if (_telemetryBatch.isNotEmpty) {
        final path = '${dir.path}/rhs_telemetry_$date.csv';
        final file = File(path);
        final exists = await file.exists();
        final sb = StringBuffer();
        if (!exists) {
          sb.writeln('timestamp,latitude,longitude,speed_kmh,impact_mag');
        }
        for (final r in _telemetryBatch) {
          sb.writeln(
              '${r['timestamp']},${r['latitude']},${r['longitude']},${r['speed_kmh']},${r['impact_mag']}');
        }
        await file.writeAsString(sb.toString(), mode: FileMode.append);
        _telemetryBatch.clear();
      }

      // Events CSV
      if (_events.isNotEmpty) {
        final path = '${dir.path}/rhs_events_$date.csv';
        final file = File(path);
        final exists = await file.exists();
        final sb = StringBuffer();
        if (!exists) {
          sb.writeln(
              'id,timestamp,latitude,longitude,speed_kmh,severity_index,raw_peak,raw_p2p,inertial_confidence,type,photo,vision_label,vision_confidence');
        }
        for (final ev in _events) {
          final j = ev.toJson();
          sb.writeln(
              '${j['id']},${j['timestamp']},${j['latitude']},${j['longitude']},${j['speed_kmh']},${j['severity_index']},${j['raw_peak']},${j['raw_p2p']},${j['inertial_confidence']},${j['type']},${_csvSafe(j['photo'])},${_csvSafe(j['vision_label'])},${j['vision_confidence']}');
        }
        await file.writeAsString(sb.toString(), mode: FileMode.write);

        // Also export GeoJSON feature collection for GIS
        final geo = await _exportGeoJson(date);
        final geoPath = '${dir.path}/rhs_events_$date.geojson';
        await File(geoPath).writeAsString(geo);

        // We do not clear _events here, because UI wants to show the session list.
      }
    } catch (e) {
      debugPrint('Flush error: $e');
    }
  }

  String _csvSafe(String? v) {
    if (v == null) return '';
    final s = v.replaceAll('"', '""');
    if (s.contains(',') || s.contains('\n')) return '"$s"';
    return s;
  }

  Future<String> _exportGeoJson(String date) async {
    final features = _events.map((e) {
      return {
        'type': 'Feature',
        'properties': e.toJson(),
        'geometry': {
          'type': 'Point',
          'coordinates': [e.longitude, e.latitude],
        },
      };
    }).toList();

    final fc = {
      'type': 'FeatureCollection',
      'name': 'RoadHealth Events $date',
      'features': features,
    };
    return jsonEncode(fc);
  }

  Future<void> exportSessionFiles() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];

      final tele = File('${dir.path}/rhs_telemetry_$date.csv');
      final ev = File('${dir.path}/rhs_events_$date.csv');
      final geo = File('${dir.path}/rhs_events_$date.geojson');

      final files = <XFile>[];
      if (await tele.exists()) files.add(XFile(tele.path));
      if (await ev.exists()) files.add(XFile(ev.path));
      if (await geo.exists()) files.add(XFile(geo.path));

      if (files.isEmpty) {
        onError?.call('No exported files found for today.');
        return;
      }

      await Share.shareXFiles(files, subject: 'Road Health Export $date');
      onSuccess?.call('Export shared');
    } catch (e) {
      onError?.call('Export failed: $e');
    }
  }

  @override
  void dispose() {
    stopLogging();
    _audioPlayer.dispose();
    cameraController?.dispose();
    _vision?.close();
    super.dispose();
  }

  String get formattedSessionDuration {
    final m = sessionDuration.inMinutes.toString().padLeft(2, '0');
    final s = (sessionDuration.inSeconds % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}

/* -----------------------------
   UI Screens (kept simple)
--------------------------------*/

class MonitorScreen extends StatefulWidget {
  final RoadHealthState appState;
  const MonitorScreen({super.key, required this.appState});

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  @override
  void initState() {
    super.initState();
    widget.appState.addListener(_onStateChange);
    widget.appState.onError = _showError;
    widget.appState.onSuccess = _showSuccess;
  }

  @override
  void dispose() {
    widget.appState.removeListener(_onStateChange);
    super.dispose();
  }

  void _onStateChange() {
    if (mounted) setState(() {});
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: RHSColors.danger),
    );
  }

  void _showSuccess(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: RHSColors.success),
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = widget.appState;

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              'Road Health',
              style: GoogleFonts.dmSans(
                fontSize: 28,
                fontWeight: FontWeight.w700,
                color: RHSColors.textPrimary,
              ),
            ),
            Text(
              'Continuous road condition intelligence',
              style: GoogleFonts.dmSans(
                fontSize: 14,
                color: RHSColors.textSecondary,
              ),
            ),
            const SizedBox(height: 24),

            // Status Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: state.isLogging ? RHSColors.primaryGradient : null,
                color: state.isLogging ? null : RHSColors.bgSecondary,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  Icon(
                    state.isLogging ? Icons.sensors : Icons.pause_circle,
                    size: 48,
                    color: state.isLogging ? Colors.white : RHSColors.textMuted,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    state.isLogging ? 'MONITORING ACTIVE' : 'PAUSED',
                    style: GoogleFonts.dmSans(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: state.isLogging ? Colors.white : RHSColors.textMuted,
                    ),
                  ),
                  if (state.isLogging) ...[
                    const SizedBox(height: 8),
                    Text(
                      '${state.formattedSessionDuration}  ${state.currentSpeedKmh.toStringAsFixed(0)} km/h',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.9),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Stats Grid
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Impact (mag)',
                    state.currentImpact.toStringAsFixed(1),
                    state.currentImpact > DetectionEngine.thresholdForSpeed(state.currentSpeedKmh)
                        ? RHSColors.danger
                        : RHSColors.primary,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Max Impact',
                    state.maxImpact.toStringAsFixed(1),
                    RHSColors.warning,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildStatCard('Detections', '${state.detections}', RHSColors.danger),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard('Photos', '${state.photosCaptured}', RHSColors.success),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Auto Mode Toggle
            GestureDetector(
              onTap: state.toggleAutoMode,
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: state.autoModeEnabled ? RHSColors.primary.withOpacity(0.1) : Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: state.autoModeEnabled ? RHSColors.primary : Colors.transparent,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.directions_car,
                      color: state.autoModeEnabled ? RHSColors.primary : RHSColors.textMuted,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Smart Auto Mode',
                            style: GoogleFonts.dmSans(fontWeight: FontWeight.w600),
                          ),
                          Text(
                            state.autoModeEnabled
                                ? 'Starts at 10 km/h, stops after 3 min idle'
                                : 'Enable hands-free operation',
                            style: GoogleFonts.dmSans(
                              fontSize: 12,
                              color: RHSColors.textSecondary,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Switch(
                      value: state.autoModeEnabled,
                      onChanged: (_) => state.toggleAutoMode(),
                      activeColor: RHSColors.primary,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Impact Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: RHSColors.primaryGradient,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'YOUR CONTRIBUTION',
                    style: GoogleFonts.dmSans(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: Colors.white.withOpacity(0.8),
                      letterSpacing: 1,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildImpactStat('${state.lifetimeDetections}', 'Detections'),
                      _buildImpactStat('${state.lifetimeKmMapped.toStringAsFixed(0)} km', 'Mapped'),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // Action Button
            SizedBox(
              height: 56,
              child: ElevatedButton(
                onPressed: state.autoModeEnabled || state.isStarting
                    ? null
                    : (state.isLogging ? state.stopLogging : state.startLogging),
                style: ElevatedButton.styleFrom(
                  backgroundColor: state.isLogging ? RHSColors.danger : RHSColors.success,
                  disabledBackgroundColor: RHSColors.bgSecondary,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: state.isStarting
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                      )
                    : Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            state.autoModeEnabled
                                ? Icons.lock
                                : (state.isLogging ? Icons.stop : Icons.play_arrow),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            state.autoModeEnabled
                                ? 'Auto Mode Active'
                                : (state.isLogging ? 'Stop Monitoring' : 'Start Monitoring'),
                            style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700),
                          ),
                        ],
                      ),
              ),
            ),

            const SizedBox(height: 12),

            Text(
              state.autoModeEnabled
                  ? (state.isLogging ? 'Analyzing road conditions...' : 'Drive to auto-start monitoring')
                  : 'Every drive improves the map',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 13,
                color: RHSColors.textSecondary,
                fontStyle: FontStyle.italic,
              ),
            ),

            const SizedBox(height: 18),

            // Recent detections (session)
            if (state.events.isNotEmpty) ...[
              Text(
                'Recent detections',
                style: GoogleFonts.dmSans(fontWeight: FontWeight.w700, color: RHSColors.textPrimary),
              ),
              const SizedBox(height: 8),
              ...state.events.take(8).map((e) => _eventTile(e)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _eventTile(DetectionEvent e) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
      child: Row(
        children: [
          Icon(Icons.warning_amber_rounded, color: RHSColors.warning),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${e.type.name}  severity ${e.severityIndex.toStringAsFixed(0)}',
                  style: GoogleFonts.dmSans(fontWeight: FontWeight.w600),
                ),
                Text(
                  '${e.timestamp.toIso8601String()}',
                  style: GoogleFonts.dmSans(fontSize: 12, color: RHSColors.textSecondary),
                ),
              ],
            ),
          ),
          if (e.photoPath != null && e.photoPath!.isNotEmpty)
            const Icon(Icons.camera_alt, color: RHSColors.primary),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: RHSColors.textSecondary)),
          const SizedBox(height: 8),
          Text(
            value,
            style: GoogleFonts.spaceMono(fontSize: 24, fontWeight: FontWeight.w700, color: color),
          ),
        ],
      ),
    );
  }

  Widget _buildImpactStat(String value, String label) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          value,
          style: GoogleFonts.spaceMono(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.dmSans(fontSize: 11, color: Colors.white.withOpacity(0.8)),
        ),
      ],
    );
  }
}

class MapScreen extends StatelessWidget {
  final RoadHealthState appState;
  const MapScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    // V2: you would render clusters here (GeoJSON -> map layer)
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.map, size: 64, color: RHSColors.primary),
            const SizedBox(height: 16),
            Text(
              'Map Coming Soon',
              style: GoogleFonts.dmSans(fontSize: 20, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(
              'V2 exports GeoJSON for GIS',
              style: GoogleFonts.dmSans(color: RHSColors.textSecondary),
            ),
          ],
        ),
      ),
    );
  }
}

class SettingsScreen extends StatefulWidget {
  final RoadHealthState appState;
  const SettingsScreen({super.key, required this.appState});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  @override
  Widget build(BuildContext context) {
    final state = widget.appState;

    return Scaffold(
      appBar: AppBar(
        title: Text('Settings', style: GoogleFonts.dmSans(fontWeight: FontWeight.w700)),
        backgroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          SwitchListTile(
            title: const Text('Audio Alerts'),
            subtitle: const Text('Play sound on detection'),
            value: state.audioEnabled,
            onChanged: (val) => setState(() => state.audioEnabled = val),
          ),
          SwitchListTile(
            title: const Text('Haptic Feedback'),
            subtitle: const Text('Vibrate on detection'),
            value: state.hapticEnabled,
            onChanged: (val) => setState(() => state.hapticEnabled = val),
          ),
          SwitchListTile(
            title: const Text('Camera Enabled'),
            subtitle: const Text('Allow camera usage'),
            value: state.cameraEnabled,
            onChanged: (val) => setState(() => state.cameraEnabled = val),
          ),
          SwitchListTile(
            title: const Text('Capture Evidence'),
            subtitle: const Text('Save photos for verification (privacy-sensitive)'),
            value: state.evidenceCaptureEnabled,
            onChanged: (val) => setState(() => state.evidenceCaptureEnabled = val),
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.upload_file),
            title: const Text('Export Session Files'),
            subtitle: const Text('Shares Telemetry CSV, Events CSV, Events GeoJSON'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () async {
              await state.exportSessionFiles();
              if (mounted) setState(() {});
            },
          ),
        ],
      ),
    );
  }
}