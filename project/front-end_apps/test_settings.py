import os
import json
import settings


class TestToggleFps:
    def test_in_range(self, monkeypatch, capsys):
        input_fps_rate = '15'
        monkeypatch.setattr('builtins.input', lambda: input_fps_rate)
        settings.toggle_fps()
        stdout, stderr = capsys.readouterr()
        assert f'FPS is set to {input_fps_rate}' in stdout

    def test_out_of_range(self, monkeypatch, capsys):
        input_fps_rate = '150'
        monkeypatch.setattr('builtins.input', lambda: input_fps_rate)
        ret = settings.toggle_fps()
        stdout, stderr = capsys.readouterr()
        assert 'ERROR: invalid option' in stdout
        assert ret == -1


class TestToggleClassMode:
    def test_single(self, monkeypatch, capsys):
        input_single_option = '1'
        monkeypatch.setattr('builtins.input', lambda: input_single_option)
        settings.toggle_classification_mode()
        stdout, stderr = capsys.readouterr()
        assert 'classification is singular' in stdout

    def test_multi(self, monkeypatch, capsys):
        input_multi_option = '2'
        monkeypatch.setattr('builtins.input', lambda: input_multi_option)
        settings.toggle_classification_mode()
        stdout, stderr = capsys.readouterr()
        assert 'classification is multi' in stdout

    def test_invalid_option(self, monkeypatch, capsys):
        input_invalid_option = '3'
        monkeypatch.setattr('builtins.input', lambda: input_invalid_option)
        ret = settings.toggle_classification_mode()
        stdout, stderr = capsys.readouterr()
        assert 'ERROR: invalid option' in stdout
        assert ret == -1


class TestToggleCompMode:
    def test_online(self, monkeypatch, capsys):
        input_online_option = '1'
        monkeypatch.setattr('builtins.input', lambda: input_online_option)
        settings.toggle_computation_mode()
        stdout, stderr = capsys.readouterr()
        assert 'Processing is online' in stdout

    def test_offline(self, monkeypatch, capsys):
        input_offline_option = '2'
        monkeypatch.setattr('builtins.input', lambda: input_offline_option)
        settings.toggle_computation_mode()
        stdout, stderr = capsys.readouterr()
        assert 'Processing is offline' in stdout

    def test_invalid_option(self, monkeypatch, capsys):
        input_invalid_option = '3'
        monkeypatch.setattr('builtins.input', lambda: input_invalid_option)
        ret = settings.toggle_computation_mode()
        stdout, stderr = capsys.readouterr()
        assert 'ERROR: invalid option' in stdout
        assert ret == -1


class TestSettingsConfig:
    def test_load(self):
        with open('temp_settings.cfg', 'w') as f:
            f.write('{"PROCESS_ONLINE": true, \
                    "SINGLE_CLASSIFICATION": true, \
                    "FPS_RATE": 10}')

        settings.load('temp_settings.cfg')
        assert settings.PROCESS_ONLINE == True
        assert settings.SINGLE_CLASSIFICATION == True
        assert settings.FPS_RATE == 10

        os.remove('temp_settings.cfg')

    def test_save(self):
        with open('temp_settings.cfg', 'w') as f:
            f.write('{"PROCESS_ONLINE": true, \
                    "SINGLE_CLASSIFICATION": true, \
                    "FPS_RATE": 10}')

        settings.load('temp_settings.cfg')
        settings.PROCESS_ONLINE = False
        settings.SINGLE_CLASSIFICATION = True
        settings.FPS_RATE = 20
        settings.save('temp_settings.cfg')

        with open('temp_settings.cfg', 'r') as f:
            contents = json.load(f)
            assert contents['PROCESS_ONLINE'] == False
            assert contents['SINGLE_CLASSIFICATION'] == True
            assert contents['FPS_RATE'] == 20

        os.remove('temp_settings.cfg')


class TestHandle:
    def test_key_invalid(self, capsys):
        invalid_option = -1
        ret = settings.handle_key(invalid_option)
        stdout, stderr = capsys.readouterr()
        assert 'ERROR: invalid option' in stdout
        assert ret == True

    def test_key_comp(self, monkeypatch, capsys):
        mock_return = 'comp'
        monkeypatch.setattr('settings.toggle_computation_mode',
                            lambda: print(mock_return))
        comp_option = settings.menu_options[0]
        ret = settings.handle_key(comp_option)
        stdout, stderr = capsys.readouterr()
        assert mock_return in stdout
        assert ret == True

    def test_key_class(self, monkeypatch, capsys):
        mock_return = 'class'
        monkeypatch.setattr(
            'settings.toggle_classification_mode', lambda: print(mock_return))
        class_option = settings.menu_options[1]
        ret = settings.handle_key(class_option)
        stdout, stderr = capsys.readouterr()
        assert mock_return in stdout
        assert ret == True

    def test_key_fps(self, monkeypatch, capsys):
        mock_return = 'fps'
        monkeypatch.setattr('settings.toggle_fps', lambda: print(mock_return))
        fps_option = settings.menu_options[2]
        ret = settings.handle_key(fps_option)
        stdout, stderr = capsys.readouterr()
        assert mock_return in stdout
        assert ret == True

    def test_key_exit(self, capsys):
        exit_option = 'M'
        ret = settings.handle_key(exit_option)
        stdout, stderr = capsys.readouterr()
        assert 'returning to menu' in stdout
        assert ret == False
