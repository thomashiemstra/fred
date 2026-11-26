// The script of the block stacking page. It asks /blocks/status every second and shows the answer, and the buttons
// call the api of the block stacking. All text from the server goes on the page with textContent, never innerHTML.
(function () {
    'use strict';

    const STATUS_URL = '/blocks/status';
    const POLL_INTERVAL = 1000;
    const NO_ANSWER = 'no answer from the server';

    function element(id) {
        return document.getElementById(id);
    }

    function oneDecimal(value) {
        return Number(value).toFixed(1);
    }

    // the answer of the server as {ok, status, data}, also when it is not json or does not come at all
    function request(method, url, payload) {
        const options = {method: method, headers: {'Accept': 'application/json'}};
        if (payload !== undefined) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(payload);
        }
        return fetch(url, options).then(function (response) {
            return response.json().then(function (data) {
                return {ok: response.ok && data.success !== false, status: response.status, data: data};
            }, function () {
                return {ok: false, status: response.status,
                        data: {error: 'unexpected answer from the server (status ' + response.status + ')'}};
            });
        }, function () {
            return {ok: false, status: 0, data: {error: NO_ANSWER}};
        });
    }

    function clear(parent) {
        while (parent.firstChild) {
            parent.removeChild(parent.firstChild);
        }
    }

    function addCell(row, text) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.appendChild(cell);
    }

    function addRow(body, texts) {
        const row = document.createElement('tr');
        texts.forEach(function (text) {
            addCell(row, text);
        });
        body.appendChild(row);
    }

    function addItem(list, text) {
        const item = document.createElement('li');
        item.textContent = text;
        list.appendChild(item);
    }

    // what a button said: the text of the error in red, or what happened
    function showAnswer(text, isError) {
        const answer = element('answer');
        answer.textContent = text;
        answer.className = 'font-weight-bold ' + (isError ? 'text-danger' : 'text-success');
    }

    // The "Click to stack" switch is a bootstrap4-toggle checkbox. The plugin fires a jQuery change event when the
    // switch is set by script as well, so the status setting it must not look like the user switching it: the
    // settingFromStatus flag tells the two apart. If jQuery or the plugin did not load (the page loads them from a
    // cdn), the plain checkbox does the same job.
    const CLICK_MODE_URL = '/blocks/click_mode';
    let settingFromStatus = false;
    let clickModeRequestPending = false;
    let lastClickMode = false;

    function clickModeToggle() {
        const jquery = window.jQuery;
        return jquery && jquery.fn && jquery.fn.bootstrapToggle ? jquery(element('clickMode')) : null;
    }

    function clickModeIsOn() {
        return element('clickMode').checked;
    }

    // shows the switch on or off, without sending a request
    function setClickModeSwitch(on) {
        if (clickModeIsOn() === on) {
            return;
        }
        settingFromStatus = true;
        try {
            const toggle = clickModeToggle();
            if (toggle) {
                toggle.bootstrapToggle(on ? 'on' : 'off', true);
            } else {
                element('clickMode').checked = on;
            }
            // the plugin keeps the checkbox in step in any case, this makes sure of it
            element('clickMode').checked = on;
        } finally {
            settingFromStatus = false;
        }
    }

    function enableClickModeSwitch(enabled) {
        const toggle = clickModeToggle();
        if (toggle) {
            toggle.bootstrapToggle(enabled ? 'enable' : 'disable');
        }
        element('clickMode').disabled = !enabled;
    }

    // the switch follows the status: it falls back to off when the mode ends by itself, and while a normal run goes
    // on it can not be used
    function renderClickMode(status) {
        const on = status.click_mode === true;
        lastClickMode = on;
        const message = element('clickMessage');
        message.textContent = status.click_message ? 'Last click: ' + status.click_message : '';
        message.hidden = !status.click_message;
        if (clickModeRequestPending) {
            // the answer to the request of the user is on its way, it ends with the switch following the status
            return;
        }
        setClickModeSwitch(on);
        enableClickModeSwitch(on || status.state === 'idle');
    }

    function onClickModeSwitched() {
        if (settingFromStatus || clickModeRequestPending) {
            return;
        }
        const wanted = clickModeIsOn();
        clickModeRequestPending = true;
        enableClickModeSwitch(false);
        showAnswer(wanted ? 'starting the click mode' : 'ending the click mode', false);
        request('POST', CLICK_MODE_URL, {on: wanted}).then(function (answer) {
            clickModeRequestPending = false;
            if (answer.ok) {
                showAnswer(wanted ? 'The click mode is on: click a block in the camera window.'
                                  : 'The click mode is off: the arm is being laid down.', false);
                enableClickModeSwitch(true);
            } else {
                showAnswer(answer.data.error || 'the request failed (status ' + answer.status + ')', true);
                // back to what the status says, the next poll corrects it if that has changed in the meantime
                setClickModeSwitch(lastClickMode);
                enableClickModeSwitch(true);
            }
        });
    }

    function bindClickModeSwitch() {
        const toggle = clickModeToggle();
        if (toggle) {
            toggle.bootstrapToggle();
            // the plugin triggers the change with jQuery, which a native listener does not hear
            toggle.on('change', onClickModeSwitched);
        } else {
            element('clickMode').addEventListener('change', onClickModeSwitched);
        }
    }

    function renderStatus(status) {
        renderClickMode(status);
        element('state').textContent = status.state;
        element('activity').textContent = status.activity ? '- ' + status.activity : '';
        element('error').textContent = status.error ? status.error : '';
        const warnings = element('warnings');
        clear(warnings);
        (status.warnings || []).forEach(function (warning) {
            addItem(warnings, 'Warning: ' + warning);
        });
        element('board').textContent = status.board_visible ? 'yes' : 'no';

        const blocks = status.blocks || [];
        const blockRows = element('blocks');
        clear(blockRows);
        blocks.forEach(function (block) {
            addRow(blockRows, [block.id, oneDecimal(block.x), oneDecimal(block.y), oneDecimal(block.yaw),
                               block.stable ? 'yes' : 'no']);
        });
        element('noBlocks').hidden = blocks.length > 0;

        const plan = status.plan || [];
        const planRows = element('plan');
        clear(planRows);
        plan.forEach(function (step, index) {
            addRow(planRows, [index + 1, step.id, step.level]);
        });
        element('noPlan').hidden = plan.length > 0;

        const skipped = element('skipped');
        clear(skipped);
        Object.keys(status.skipped || {}).forEach(function (id) {
            addItem(skipped, 'block ' + id + ': ' + status.skipped[id]);
        });

        const tower = status.tower || [];
        element('tower').textContent = tower.length > 0
            ? tower.join(', ') + ' (bottom first)' : 'no block is stacked yet.';
    }

    // the status could not be read: say so where the error goes, the rest of the page stays as it was
    function renderNoStatus(message) {
        element('state').textContent = 'unknown';
        element('error').textContent = message;
    }

    function poll() {
        request('GET', STATUS_URL).then(function (answer) {
            if (answer.ok) {
                renderStatus(answer.data);
            } else {
                renderNoStatus(answer.data.error || NO_ANSWER);
            }
            window.setTimeout(poll, POLL_INTERVAL);
        });
    }

    function renderTip(tip) {
        const result = element('tipResult');
        const x = tip.tip[0], y = tip.tip[1], z = tip.tip[2];
        let text = 'The tip of the arm is at x ' + oneDecimal(x) + ', y ' + oneDecimal(y) + ', z ' + oneDecimal(z) +
            ' cm. ';
        if (tip.nearest_block === null || tip.nearest_block === undefined) {
            text += 'No block is in the picture to compare it with.';
        } else {
            text += 'The nearest block is ' + tip.nearest_block + ', at x ' + oneDecimal(tip.block[0]) + ', y ' +
                oneDecimal(tip.block[1]) + ' cm: the block minus the tip is x ' + oneDecimal(tip.difference[0]) +
                ', y ' + oneDecimal(tip.difference[1]) + ' cm.';
        }
        result.textContent = text;
    }

    // sends the request of a button, the button stays disabled until the answer is there (a look takes seconds)
    function act(button, url, payload, pending, done, onSuccess) {
        button.disabled = true;
        showAnswer(pending, false);
        request('POST', url, payload).then(function (answer) {
            button.disabled = false;
            if (answer.ok) {
                showAnswer(done, false);
                if (onSuccess) {
                    onSuccess(answer.data);
                }
            } else {
                showAnswer(answer.data.error || 'the request failed (status ' + answer.status + ')', true);
            }
        });
    }

    function bind(id, url, payload, pending, done, onSuccess, confirmation) {
        const button = element(id);
        button.addEventListener('click', function () {
            if (confirmation && !window.confirm(confirmation)) {
                return;
            }
            act(button, url, payload, pending, done, onSuccess);
        });
    }

    function bindButtons() {
        bind('cameraOn', '/blocks/camera/start', undefined, 'starting the camera', 'The camera is on.');
        bind('cameraOff', '/blocks/camera/stop', undefined, 'stopping the camera', 'The camera is off.');
        bind('look', '/blocks/scan', undefined, 'looking at the table', 'Looked at the table.', renderStatus);
        bind('checkTip', '/blocks/tip', undefined, 'reading the tip', 'Read the tip.', renderTip);
        bind('rehearse', '/blocks/start', {hover_only: true}, 'starting the rehearsal', 'The rehearsal has started.');
        bind('stack', '/blocks/start', {hover_only: false}, 'starting to stack', 'Stacking has started.');
        bind('stop', '/blocks/stop', undefined, 'stopping', 'Stopping: the arm is laid down on the table.');
        bind('layDown', '/blocks/lay_down', undefined, 'laying the arm down', 'The arm is being laid down.');
        bind('torqueOff', '/blocks/disable_servos', undefined, 'switching the torque off', 'The torque is off.',
             undefined, 'The arm drops when it is in the air. Switch the torque off?');
    }

    bindButtons();
    bindClickModeSwitch();
    poll();
})();
