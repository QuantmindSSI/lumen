import * as vscode from 'vscode';

/**
 * Minimal VS Code extension that registers a Copilot Chat participant
 * for Lumen memory. Requires VS Code 1.90+ with chat participant API.
 */
export function activate(context: vscode.ExtensionContext) {
  const participant = vscode.chat.createChatParticipant(
    'copilot.lumen',
    async (request, _context, response, _token) => {
      const query = request.prompt;

      // Assemble context from Lumen
      try {
        const res = await fetch('http://localhost:8848/assemble', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: 5 }),
        });
        if (!res.ok) {
          response.markdown(`Lumen server error: ${res.statusText}. Is \\"lumen serve\\" running?`);
          return;
        }
        const data = await res.json();
        if (data.assembled_context) {
          response.markdown(`**Lumen context retrieved:**\n\n${data.assembled_context}`);
        } else {
          response.markdown('No relevant memories found in the Lumen palace.');
        }
      } catch (err) {
        response.markdown(
          `Failed to reach Lumen server at localhost:8848. ` +
          `Please start it with: \`lumen serve\``
        );
      }
    }
  );

  participant.iconPath = new vscode.ThemeIcon('database');
  participant.description = 'Lumen Memory Palace';

  context.subscriptions.push(participant);
}
