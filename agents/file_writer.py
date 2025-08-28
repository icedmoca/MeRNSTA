#!/usr/bin/env python3
"""
FileWriter Agent for MeRNSTA - Phase 14: Recursive Execution

Autonomous code file generation with safety checks, metadata, and versioning.
Supports writing executable scripts, Python modules, and other code files.
"""

import os
import stat
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from config.settings import get_config


class FileWriter:
    """
    Autonomous file writer with safety checks and metadata generation.
    
    Features:
    - Safe file creation with configurable directories
    - Automatic metadata headers
    - Version suffixes and timestamps
    - File type detection and appropriate headers
    - Executable permission management
    - Collision detection and resolution
    """
    
    def __init__(self):
        self.config = get_config().get('recursive_execution', {})
        self.write_dir = Path(self.config.get('write_dir', './generated/'))
        self.safe_extensions = self.config.get('safe_extensions', ['.py', '.sh', '.js', '.sql', '.json', '.yaml', '.yml', '.txt', '.md'])
        self.enable_versioning = self.config.get('enable_versioning', True)
        self.add_metadata = self.config.get('add_metadata', True)
        
        # Ensure write directory exists
        self.write_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger('file_writer')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"[FileWriter] Initialized with write_dir={self.write_dir}")
    
    def write_file(self, content: str, filename: str, 
                   directory: Optional[Union[str, Path]] = None,
                   executable: bool = False,
                   force_overwrite: bool = False,
                   add_timestamp: bool = True,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Write content to a file with safety checks and metadata.
        
        Args:
            content: File content to write
            filename: Target filename
            directory: Target directory (defaults to write_dir)
            executable: Whether to make file executable
            force_overwrite: Whether to overwrite existing files
            add_timestamp: Whether to add timestamp to filename
            metadata: Additional metadata to include in header
            
        Returns:
            Result dictionary with success status, file path, and metadata
        """
        try:
            # Determine target directory
            target_dir = Path(directory) if directory else self.write_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.safe_extensions:
                return self._create_result(
                    success=False,
                    error=f"Unsafe file extension: {file_ext}. Allowed: {self.safe_extensions}",
                    filename=filename
                )
            
            # Generate final filename with versioning
            final_filename = self._generate_filename(
                filename, target_dir, add_timestamp, force_overwrite
            )
            final_path = target_dir / final_filename
            
            # Add metadata header if enabled
            if self.add_metadata:
                content = self._add_metadata_header(content, file_ext, metadata or {})
            
            # Write file
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Set executable permissions if requested
            if executable:
                self._make_executable(final_path)
            
            # Calculate file hash for verification
            file_hash = self._calculate_file_hash(final_path)
            
            self.logger.info(f"[FileWriter] Successfully wrote file: {final_path}")
            
            return self._create_result(
                success=True,
                filepath=str(final_path),
                filename=final_filename,
                size=final_path.stat().st_size,
                hash=file_hash,
                executable=executable,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"[FileWriter] Error writing file {filename}: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                filename=filename
            )
    
    def write_python_script(self, code: str, script_name: str,
                          imports: Optional[List[str]] = None,
                          main_function: bool = True,
                          **kwargs) -> Dict[str, Any]:
        """
        Write a Python script with proper structure and imports.
        
        Args:
            code: Python code content
            script_name: Script filename (without .py extension)
            imports: List of import statements to add
            main_function: Whether to wrap code in if __name__ == '__main__':
            **kwargs: Additional arguments for write_file
            
        Returns:
            Result dictionary from write_file
        """
        # Ensure .py extension
        if not script_name.endswith('.py'):
            script_name += '.py'
        
        # Build script content
        script_content = []
        
        # Add shebang for Python scripts
        script_content.append('#!/usr/bin/env python3')
        script_content.append('')
        
        # Add imports
        if imports:
            for imp in imports:
                if not imp.strip().startswith(('import ', 'from ')):
                    imp = f'import {imp}'
                script_content.append(imp)
            script_content.append('')
        
        # Add main code
        if main_function and 'if __name__' not in code:
            script_content.append('def main():')
            # Indent the code
            indented_code = '\n'.join(f'    {line}' for line in code.split('\n'))
            script_content.append(indented_code)
            script_content.append('')
            script_content.append('if __name__ == "__main__":')
            script_content.append('    main()')
        else:
            script_content.append(code)
        
        final_content = '\n'.join(script_content)
        
        return self.write_file(
            content=final_content,
            filename=script_name,
            executable=True,
            **kwargs
        )
    
    def write_shell_script(self, commands: Union[str, List[str]], script_name: str,
                          shell: str = '/bin/bash', **kwargs) -> Dict[str, Any]:
        """
        Write a shell script with proper structure.
        
        Args:
            commands: Shell commands (string or list)
            script_name: Script filename (without .sh extension)
            shell: Shell interpreter path
            **kwargs: Additional arguments for write_file
            
        Returns:
            Result dictionary from write_file
        """
        # Ensure .sh extension
        if not script_name.endswith('.sh'):
            script_name += '.sh'
        
        # Build script content
        script_content = [f'#!{shell}', '']
        
        if isinstance(commands, list):
            script_content.extend(commands)
        else:
            script_content.append(commands)
        
        final_content = '\n'.join(script_content)
        
        return self.write_file(
            content=final_content,
            filename=script_name,
            executable=True,
            **kwargs
        )
    
    def list_generated_files(self, directory: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        List all generated files in the target directory.
        
        Args:
            directory: Directory to list (defaults to write_dir)
            
        Returns:
            Dictionary with file list and metadata
        """
        try:
            target_dir = Path(directory) if directory else self.write_dir
            
            if not target_dir.exists():
                return {'success': True, 'files': [], 'directory': str(target_dir)}
            
            files = []
            for file_path in target_dir.iterdir():
                if file_path.is_file():
                    stat_info = file_path.stat()
                    files.append({
                        'name': file_path.name,
                        'path': str(file_path),
                        'size': stat_info.st_size,
                        'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        'executable': bool(stat_info.st_mode & stat.S_IEXEC),
                        'extension': file_path.suffix
                    })
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            return {
                'success': True,
                'files': files,
                'directory': str(target_dir),
                'count': len(files)
            }
            
        except Exception as e:
            self.logger.error(f"[FileWriter] Error listing files: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_file(self, filename: str, directory: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Safely delete a generated file.
        
        Args:
            filename: File to delete
            directory: Directory containing file (defaults to write_dir)
            
        Returns:
            Result dictionary with success status
        """
        try:
            target_dir = Path(directory) if directory else self.write_dir
            file_path = target_dir / filename
            
            if not file_path.exists():
                return {'success': False, 'error': f'File not found: {filename}'}
            
            # Safety check: ensure file is in allowed directory
            if not str(file_path.resolve()).startswith(str(target_dir.resolve())):
                return {'success': False, 'error': 'File outside allowed directory'}
            
            file_path.unlink()
            self.logger.info(f"[FileWriter] Deleted file: {file_path}")
            
            return {'success': True, 'deleted_file': str(file_path)}
            
        except Exception as e:
            self.logger.error(f"[FileWriter] Error deleting file {filename}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_filename(self, filename: str, target_dir: Path, 
                          add_timestamp: bool, force_overwrite: bool) -> str:
        """Generate final filename with versioning and collision detection."""
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        
        if force_overwrite or not (target_dir / filename).exists():
            return filename
        
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{base_name}_{timestamp}{extension}"
        
        # Find next available version number
        counter = 1
        while True:
            versioned_name = f"{base_name}_v{counter}{extension}"
            if not (target_dir / versioned_name).exists():
                return versioned_name
            counter += 1
    
    def _add_metadata_header(self, content: str, file_ext: str, metadata: Dict[str, Any]) -> str:
        """Add metadata header based on file type."""
        timestamp = datetime.now().isoformat()
        header_lines = []
        
        # Determine comment style
        if file_ext in ['.py']:
            comment_char = '#'
        elif file_ext in ['.sh']:
            comment_char = '#'
        elif file_ext in ['.js']:
            comment_char = '//'
        elif file_ext in ['.sql']:
            comment_char = '--'
        else:
            comment_char = '#'
        
        # Build header
        header_lines.append(f"{comment_char} Autogenerated by MeRNSTA FileWriter")
        header_lines.append(f"{comment_char} Generated: {timestamp}")
        
        if metadata:
            header_lines.append(f"{comment_char} Metadata:")
            for key, value in metadata.items():
                header_lines.append(f"{comment_char}   {key}: {value}")
        
        header_lines.append(f"{comment_char} " + "="*50)
        header_lines.append("")
        
        return '\n'.join(header_lines) + content
    
    def _make_executable(self, file_path: Path):
        """Make file executable."""
        current_permissions = file_path.stat().st_mode
        file_path.chmod(current_permissions | stat.S_IEXEC)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_result(self, success: bool, **kwargs) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        result = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        result.update(kwargs)
        return result


# Global file writer instance
_file_writer = None

def get_file_writer() -> FileWriter:
    """Get or create global file writer instance."""
    global _file_writer
    if _file_writer is None:
        _file_writer = FileWriter()
    return _file_writer


def write_code_file(content: str, filename: str, **kwargs) -> Dict[str, Any]:
    """
    Convenient wrapper for writing code files.
    
    Args:
        content: File content
        filename: Target filename
        **kwargs: Additional arguments for FileWriter.write_file
        
    Returns:
        Result dictionary
    """
    writer = get_file_writer()
    return writer.write_file(content, filename, **kwargs)


def write_and_execute(content: str, filename: str, executor_name: str = "file_writer") -> Dict[str, Any]:
    """
    Write a file and immediately execute it.
    
    Args:
        content: File content
        filename: Target filename
        executor_name: Name to use for command execution
        
    Returns:
        Combined result with write and execution results
    """
    from agents.command_router import get_command_router
    
    # Write the file
    write_result = write_code_file(content, filename, executable=True)
    
    if not write_result['success']:
        return {
            'success': False,
            'error': f"Failed to write file: {write_result.get('error')}",
            'write_result': write_result
        }
    
    # Execute the file
    file_path = write_result['filepath']
    
    # Determine execution command based on file type
    if filename.endswith('.py'):
        command = f'/run_shell "python3 {file_path}"'
    elif filename.endswith('.sh'):
        command = f'/run_shell "bash {file_path}"'
    else:
        command = f'/run_shell "{file_path}"'
    
    # Execute through command router
    router = get_command_router()
    import asyncio
    
    try:
        # Handle async execution
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a task if in async context
            execution_result = asyncio.create_task(router.execute_command(command, executor_name))
            execution_result = asyncio.run(execution_result)
        else:
            execution_result = loop.run_until_complete(router.execute_command(command, executor_name))
    except RuntimeError:
        # No event loop running
        execution_result = asyncio.run(router.execute_command(command, executor_name))
    
    return {
        'success': write_result['success'] and execution_result['success'],
        'write_result': write_result,
        'execution_result': execution_result,
        'filepath': file_path,
        'command': command
    }