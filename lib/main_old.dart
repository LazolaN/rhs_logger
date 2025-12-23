import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:share_plus/share_plus.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';

// Global camera list
List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Error initializing cameras: $e');
  }
  
  runApp(const RHSLoggerApp());
}

class RHSLoggerApp extends StatelessWidget {
  const RHSLoggerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Health',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2196F3),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        cardTheme: const CardThemeData(
          elevation: 4,
        ),
      ),
      home: const LoggerHomePage(),
    );
  }
}

class LoggerHomePage extends StatefulWidget {
  const LoggerHomePage({super.key});

  @override
  State<LoggerHomePage> createState() => _LoggerHomePageState();
}

class _LoggerHomePageState extends State<LoggerHomePage> with WidgetsBindingObserver {
  bool _isLogging = false;
  double _currentZForce = 0.0;
  double _maxZForce = 0.0;
  int _totalEventsLogged = 0;
  int _potholesDetected = 0;
  int _photosCaptured = 0;
  DateTime? _sessionStartTime;
  
  StreamSubscription<UserAccelerometerEvent>? _accelerometerSubscription;
  Timer? _gpsTimer;
  Timer? _batchSaveTimer;
  
  final List<Map<String, dynamic>> _dataBatch = [];
  final List<String> _capturedPhotos = [];
  Position? _currentPosition;
  
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _audioEnabled = true;
  bool _cameraEnabled = true;
  
  CameraController? _cameraController;
  ImageLabeler? _imageLabeler;
  bool _isCapturing = false;
  
  static const double POTHOLE_THRESHOLD = 15.0;
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initForegroundTask();
    _initializeCamera();
    _initializeMLKit();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopLogging();
    _audioPlayer.dispose();
    _cameraController?.dispose();
    _imageLabeler?.close();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      debugPrint('No cameras available');
      return;
    }
    
    try {
      _cameraController = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
      );
      
      await _cameraController!.initialize();
      setState(() {});
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _initializeMLKit() async {
    try {
      final options = ImageLabelerOptions(confidenceThreshold: 0.5);
      _imageLabeler = ImageLabeler(options: options);
    } catch (e) {
      debugPrint('Error initializing ML Kit: $e');
    }
  }

  void _initForegroundTask() {
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'rhs_logger_channel',
        channelName: 'RHS Logger Service',
        channelDescription: 'Road pothole detection service',
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

  Future<bool> _requestPermissions() async {
    // Location permission
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        _showError('Location permission denied');
        return false;
      }
    }
    
    if (permission == LocationPermission.deniedForever) {
      _showError('Location permission permanently denied');
      return false;
    }

    // Notification permission
    if (Platform.isAndroid) {
      final notificationStatus = await Permission.notification.request();
      if (!notificationStatus.isGranted) {
        _showError('Notification permission denied');
        return false;
      }
    }
    
    // Camera permission
    final cameraStatus = await Permission.camera.request();
    if (!cameraStatus.isGranted) {
      _showError('Camera permission needed for photo capture');
      _cameraEnabled = false;
    }

    return true;
  }

  Future<void> _startLogging() async {
    if (_isLogging) return;
    
    final hasPermissions = await _requestPermissions();
    if (!hasPermissions) return;

    if (await FlutterForegroundTask.isRunningService) {
      await FlutterForegroundTask.restartService();
    } else {
      await FlutterForegroundTask.startService(
        notificationTitle: 'Road Health Monitoring',
        notificationText: 'Analyzing road conditions...',
      );
    }

    setState(() {
      _isLogging = true;
      _totalEventsLogged = 0;
      _potholesDetected = 0;
      _photosCaptured = 0;
      _maxZForce = 0.0;
      _sessionStartTime = DateTime.now();
      _dataBatch.clear();
      _capturedPhotos.clear();
    });

    _accelerometerSubscription = userAccelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 20),
    ).listen((UserAccelerometerEvent event) {
      setState(() {
        _currentZForce = event.z;
        if (event.z.abs() > _maxZForce) {
          _maxZForce = event.z.abs();
        }
      });
      
      _logAccelerometerData(event);
    });

    _gpsTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _updateGPSData();
    });

    _batchSaveTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      _saveBatchToCSV();
    });

    _showSuccess('Logging started with camera detection');
  }

  Future<void> _stopLogging() async {
    if (!_isLogging) return;

    await _accelerometerSubscription?.cancel();
    _accelerometerSubscription = null;
    
    _gpsTimer?.cancel();
    _gpsTimer = null;
    
    _batchSaveTimer?.cancel();
    _batchSaveTimer = null;

    await _saveBatchToCSV();
    await FlutterForegroundTask.stopService();

    setState(() {
      _isLogging = false;
    });

    _showSuccess('Session complete! Events: $_totalEventsLogged, Photos: $_photosCaptured');
  }

  void _logAccelerometerData(UserAccelerometerEvent event) {
    if (_currentPosition == null) return;

    final timestamp = DateTime.now().toIso8601String();
    
    _dataBatch.add({
      'timestamp': timestamp,
      'latitude': _currentPosition!.latitude,
      'longitude': _currentPosition!.longitude,
      'speed': _currentPosition!.speed,
      'accel_x': event.x,
      'accel_y': event.y,
      'accel_z': event.z,
      'photo': '',
    });

    setState(() {
      _totalEventsLogged++;
    });

    final magnitude = event.z.abs();
    if (magnitude > POTHOLE_THRESHOLD) {
      _onPotholeDetected(magnitude);
    }
  }

  Future<void> _onPotholeDetected(double magnitude) async {
    setState(() {
      _potholesDetected++;
    });
    
    if (_audioEnabled) {
      _playBeep();
    }
    
    // Trigger camera capture
    if (_cameraEnabled && _cameraController != null && _cameraController!.value.isInitialized && !_isCapturing) {
      await _captureAndAnalyzePhoto(magnitude);
    }
    
    debugPrint('Pothole detected! Magnitude: $magnitude m/s²');
  }

  Future<void> _captureAndAnalyzePhoto(double magnitude) async {
    if (_isCapturing) return;
    
    setState(() {
      _isCapturing = true;
    });
    
    try {
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final imagePath = '${directory.path}/pothole_$timestamp.jpg';
      
      final image = await _cameraController!.takePicture();
      await File(image.path).copy(imagePath);
      
      // Analyze image with ML Kit
      final inputImage = InputImage.fromFilePath(imagePath);
      final labels = await _imageLabeler?.processImage(inputImage);
      
      String classification = 'pothole'; // Default
      double confidence = 0.0;
      
      if (labels != null && labels.isNotEmpty) {
        // Check for keywords that might indicate road features
        for (var label in labels) {
          debugPrint('ML Label: ${label.label} (${label.confidence})');
          
          // Simple keyword matching
          final lowerLabel = label.label.toLowerCase();
          if (lowerLabel.contains('bump') || lowerLabel.contains('speed')) {
            classification = 'speed_bump';
            confidence = label.confidence;
            break;
          } else if (lowerLabel.contains('road') || lowerLabel.contains('asphalt') || lowerLabel.contains('pavement')) {
            classification = magnitude > 25 ? 'pothole' : 'rough_road';
            confidence = label.confidence;
          }
        }
      }
      
      // Update last entry in batch with photo info
      if (_dataBatch.isNotEmpty) {
        _dataBatch.last['photo'] = imagePath;
        _dataBatch.last['classification'] = classification;
        _dataBatch.last['confidence'] = confidence;
      }
      
      _capturedPhotos.add(imagePath);
      
      setState(() {
        _photosCaptured++;
      });
      
      debugPrint('Photo captured and analyzed: $classification ($confidence)');
      
    } catch (e) {
      debugPrint('Error capturing photo: $e');
    } finally {
      setState(() {
        _isCapturing = false;
      });
    }
  }

  Future<void> _playBeep() async {
    try {
      await _audioPlayer.play(AssetSource('sounds/beep.mp3'));
    } catch (e) {
      debugPrint('Audio playback failed: $e');
    }
  }

  Future<void> _updateGPSData() async {
    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      _currentPosition = position;
    } catch (e) {
      debugPrint('GPS error: $e');
    }
  }

  Future<void> _saveBatchToCSV() async {
    if (_dataBatch.isEmpty) return;

    try {
      final directory = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];
      final filePath = '${directory.path}/rhs_logger_$date.csv';
      final file = File(filePath);

      final fileExists = await file.exists();
      final buffer = StringBuffer();
      
      if (!fileExists) {
        buffer.writeln('timestamp,latitude,longitude,speed,accel_x,accel_y,accel_z,photo,classification,confidence');
      }

      for (final data in _dataBatch) {
        buffer.writeln(
          '${data['timestamp']},'
          '${data['latitude']},'
          '${data['longitude']},'
          '${data['speed']},'
          '${data['accel_x']},'
          '${data['accel_y']},'
          '${data['accel_z']},'
          '${data['photo'] ?? ''},'
          '${data['classification'] ?? ''},'
          '${data['confidence'] ?? ''}'
        );
      }

      await file.writeAsString(buffer.toString(), mode: FileMode.append);
      debugPrint('Saved ${_dataBatch.length} records to $filePath');
      
      _dataBatch.clear();
    } catch (e) {
      debugPrint('Error saving batch: $e');
    }
  }

  Future<void> _exportCSV() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final date = DateTime.now().toIso8601String().split('T')[0];
      final filePath = '${directory.path}/rhs_logger_$date.csv';
      final file = File(filePath);

      if (!await file.exists()) {
        _showError('No data file found for today');
        return;
      }

      await Share.shareXFiles(
        [XFile(filePath)],
        subject: 'RHS Logger Data - $date',
        text: 'Road Health Score data with photos collected on $date',
      );
      
      _showSuccess('CSV file ready to share!');
    } catch (e) {
      _showError('Export failed: $e');
    }
  }

  void _viewPhotos() {
    if (_capturedPhotos.isEmpty) {
      _showError('No photos captured yet');
      return;
    }
    
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PhotoGalleryPage(photos: _capturedPhotos),
      ),
    );
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _showSuccess(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  String _getSessionDuration() {
    if (_sessionStartTime == null || !_isLogging) return '00:00';
    final duration = DateTime.now().difference(_sessionStartTime!);
    final minutes = duration.inMinutes.toString().padLeft(2, '0');
    final seconds = (duration.inSeconds % 60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text(
          'Road Health',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(_audioEnabled ? Icons.volume_up : Icons.volume_off),
            onPressed: () {
              setState(() {
                _audioEnabled = !_audioEnabled;
              });
              _showSuccess(_audioEnabled ? 'Audio alerts ON' : 'Audio alerts OFF');
            },
          ),
          IconButton(
            icon: Icon(_cameraEnabled ? Icons.camera_alt : Icons.camera_alt_outlined),
            onPressed: () {
              setState(() {
                _cameraEnabled = !_cameraEnabled;
              });
              _showSuccess(_cameraEnabled ? 'Camera detection ON' : 'Camera detection OFF');
            },
          ),
          IconButton(
            icon: const Icon(Icons.photo_library),
            onPressed: _viewPhotos,
            tooltip: 'View Photos',
          ),
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _exportCSV,
            tooltip: 'Export CSV',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // Logo Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      Image.asset(
                        'assets/images/rhs_logo.png',
                        height: 100,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Road Health Score',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF2196F3),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'AI-Powered Detection',
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Status Card
              Card(
                color: _isLogging ? Colors.green[50] : Colors.grey[50],
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Column(
                        children: [
                          Icon(
                            _isLogging ? Icons.fiber_manual_record : Icons.stop_circle,
                            color: _isLogging ? Colors.green : Colors.grey,
                            size: 32,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _isLogging ? 'RECORDING' : 'STOPPED',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: _isLogging ? Colors.green : Colors.grey,
                            ),
                          ),
                        ],
                      ),
                      if (_isLogging)
                        Column(
                          children: [
                            const Icon(Icons.timer, color: Colors.blue, size: 32),
                            const SizedBox(height: 8),
                            Text(
                              _getSessionDuration(),
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 18,
                              ),
                            ),
                          ],
                        ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Statistics Grid
              GridView.count(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 2,
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: 1.2,
                children: [
                  _buildStatCard(
                    'Road Impact',
                    '${_currentZForce.toStringAsFixed(1)} m/s²',
                    Icons.trending_up,
                    _currentZForce.abs() > POTHOLE_THRESHOLD 
                        ? Colors.red 
                        : Colors.blue,
                  ),
                  _buildStatCard(
                    'Max Impact',
                    '${_maxZForce.toStringAsFixed(1)} m/s²',
                    Icons.vertical_align_top,
                    Colors.orange,
                  ),
                  _buildStatCard(
                    'Detections',
                    _potholesDetected.toString(),
                    Icons.warning,
                    Colors.red,
                  ),
                  _buildStatCard(
                    'Photos',
                    _photosCaptured.toString(),
                    Icons.camera,
                    Colors.purple,
                  ),
                ],
              ),
              
              const SizedBox(height: 16),
              
              // Camera & GPS Status
              Card(
                child: Column(
                  children: [
                    ListTile(
                      leading: Icon(
                        _cameraController?.value.isInitialized ?? false 
                            ? Icons.camera_alt 
                            : Icons.camera_alt_outlined,
                        color: _cameraController?.value.isInitialized ?? false 
                            ? Colors.green 
                            : Colors.grey,
                      ),
                      title: Text(
                        _cameraController?.value.isInitialized ?? false 
                            ? 'Camera Ready' 
                            : 'Camera Initializing...',
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                      ),
                      subtitle: Text(
                        _cameraEnabled ? 'Auto-capture enabled' : 'Auto-capture disabled',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                    const Divider(height: 1),
                    ListTile(
                      leading: Icon(
                        _currentPosition != null ? Icons.gps_fixed : Icons.gps_not_fixed,
                        color: _currentPosition != null ? Colors.green : Colors.grey,
                      ),
                      title: Text(
                        _currentPosition != null ? 'GPS Active' : 'Waiting for GPS...',
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                      ),
                      subtitle: _currentPosition != null
                          ? Text(
                              'Speed: ${(_currentPosition!.speed * 3.6).toStringAsFixed(1)} km/h',
                              style: const TextStyle(fontSize: 12),
                            )
                          : const Text('Acquiring location', style: TextStyle(fontSize: 12)),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Main Action Button
              SizedBox(
                width: double.infinity,
                height: 60,
                child: ElevatedButton(
                  onPressed: _isLogging ? _stopLogging : _startLogging,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isLogging ? Colors.red : Colors.green,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                    elevation: 8,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        _isLogging ? Icons.stop : Icons.play_arrow,
                        size: 32,
                      ),
                      const SizedBox(width: 12),
                      Text(
                        _isLogging ? 'Stop Logging' : 'Start Logging',
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              Text(
                _isLogging 
                    ? 'AI analyzing road conditions...'
                    : 'Press Start to begin intelligent road monitoring',
                style: TextStyle(
                  color: Colors.grey[600],
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon, Color color) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 28, color: color),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

// Photo Gallery Page
class PhotoGalleryPage extends StatelessWidget {
  final List<String> photos;
  
  const PhotoGalleryPage({super.key, required this.photos});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Captured Photos (${photos.length})'),
        elevation: 0,
      ),
      body: GridView.builder(
        padding: const EdgeInsets.all(8),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          crossAxisSpacing: 8,
          mainAxisSpacing: 8,
        ),
        itemCount: photos.length,
        itemBuilder: (context, index) {
          return GestureDetector(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => FullScreenImage(imagePath: photos[index]),
                ),
              );
            },
            child: Card(
              clipBehavior: Clip.antiAlias,
              child: Image.file(
                File(photos[index]),
                fit: BoxFit.cover,
              ),
            ),
          );
        },
      ),
    );
  }
}

// Full Screen Image Viewer
class FullScreenImage extends StatelessWidget {
  final String imagePath;
  
  const FullScreenImage({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Center(
        child: InteractiveViewer(
          child: Image.file(File(imagePath)),
        ),
      ),
    );
  }
}