/* SPDX-FileCopyrightText: 2026 aimcrazyxx
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * LD_PRELOAD test double for the GUI check only (kaggle/build_mixar_linux.sh).
 *
 * Release builds refuse to open their UI until a sign-in token is stored in
 * the Secret Service (the startup SSO gate in source/creator), and a headless
 * check can neither sign in through a browser nor rely on a desktop keyring.
 * This answers the gate's libsecret lookup with a placeholder so the UI can
 * load. The placeholder is not a credential: the backend rejects it, and the
 * app's Python side, which uses its own keyring client, still sees no token.
 */

#include <stdlib.h>
#include <string.h>

char *secret_password_lookup_sync(const void *schema, void *cancellable, void **error, ...)
{
  (void)schema;
  (void)cancellable;
  if (error) {
    *error = NULL;
  }
  return strdup("gui-check-placeholder");
}

void secret_password_free(char *password)
{
  free(password);
}
