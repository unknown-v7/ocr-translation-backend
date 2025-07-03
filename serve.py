from waitress import serve
import ocr_api_server
serve(ocr_api_server.app, host='0.0.0.0', port=5000)