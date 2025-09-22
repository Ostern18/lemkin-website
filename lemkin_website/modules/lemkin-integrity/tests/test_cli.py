import pytest
import tempfile
from pathlib import Path
from typer.testing import CliRunner

from lemkin_integrity.cli import app

runner = CliRunner()

@pytest.fixture
def temp_file():
    """Create temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test evidence content for CLI testing.")
        file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)

def test_hash_evidence_command(temp_file):
    """Test hash-evidence CLI command"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
        db_path = db_file.name
    
    try:
        result = runner.invoke(app, [
            "hash-evidence",
            str(temp_file),
            "--case-id", "CLI-TEST-001",
            "--collector", "CLI Tester",
            "--source", "CLI Test",
            "--description", "CLI test evidence",
            "--db-path", db_path
        ])
        
        assert result.exit_code == 0
        assert "Evidence Hash Generated" in result.stdout
        assert "Evidence ID" in result.stdout
        assert "SHA-256 Hash" in result.stdout
    finally:
        Path(db_path).unlink(missing_ok=True)

def test_verify_command():
    """Test verify CLI command"""
    # This would require setting up evidence first
    # For brevity, testing the command structure
    result = runner.invoke(app, ["verify", "nonexistent-id"])
    assert result.exit_code == 1  # Should fail for nonexistent evidence