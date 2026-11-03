# GitHub upload and an anonymous review mirror

## Current local status

This artifact is prepared locally. No remote GitHub repository or Anonymous GitHub mirror has been created automatically. Original-contribution MIT coverage does not resolve the unconfirmed redistribution permission for mixed Meta-IFD components; keep the repository private while those terms are being clarified.

## Upload using GitHub Desktop

1. Install [GitHub Desktop](https://desktop.github.com/) and sign in to your GitHub account.
2. Choose **File > Add local repository** and select the prepared local Git working repository. If using an extracted ZIP without a `.git` directory, use Desktop's option to create a repository in that existing folder without replacing its contents.
3. Confirm that `README.md`, `code`, `configs`, `results`, `figures`, `docs`, `provenance`, the two license files and the hidden `.github`, `.gitignore` and `.gitattributes` files are included. `README.md` belongs at the repository root, not inside another nested archive folder.
4. Commit the initial files with a message such as `Add SIFE44 research artifact`.
5. Select **Publish repository**, use a name such as `SIFE44-Ethereum-Fraud-Detection`, and initially leave **Keep this code private** selected.
6. Open **View on GitHub**. The README should render as the project homepage. The workflow under **Actions** verifies the artifact when Actions is enabled and available to the repository.

Upload the actual folder contents, not just the ZIP file. A ZIP uploaded as a single file does not become a browsable repository. This project has hundreds of files; GitHub Desktop/Git is more convenient than the browser's limited batch uploader.

Official instructions: [add a local repository](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-a-repository-from-your-local-computer-to-github-desktop) and [publish an existing project](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop).

## Create the anonymous mirror

1. Open [Anonymous GitHub](https://anonymous.4open.science/) and sign in with GitHub. Prefer the service's GitHub App connection, granting read access only to the intended repository.
2. Open the anonymization form, enter the source repository URL and select the intended branch/commit if the interface offers that choice. Source accessibility depends on the account connection; do not assume a private repository is accessible without an explicit read grant.
3. Enter a neutral review name and add the authors' names, institution, emails and any source-account identifiers to **Terms to anonymize**, one regular expression per line. The account/repository name can also be handled by the service. Match whole identifiers, not broad fragments such as `Tran` that might alter `transfer` in scientific code.
4. Check split author-name fields in `CITATION.cff` as well as full names in `README.md`, `LICENSE` and `CODE_LICENSE_SCOPE.json`. A full-name replacement alone does not hide separately stored given/family names.
5. Select an expiration after the review period. If preserving anonymity matters, avoid an expiry policy that redirects reviewers to the original named repository during review.
6. Enable the file types needed by reviewers. Review image/PDF contents and metadata yourself: the service serves binary files unchanged rather than removing identities inside them. The five supplied figure exports were locally checked for the supplied author/institution identifiers; this is not a guarantee against every possible identifier.
7. Generate the mirror and open its URL in a signed-out/private window. Check README, citation metadata, license, provenance, notebooks and the download option. Use the anonymous URL in review materials, not the original named GitHub URL.

The author term list is supplied separately from the repository to avoid publishing a ready-made identity list inside the mirror.

## Size and evidence integrity

The service FAQ states that files larger than 8 MB are unsupported. Main TEST prediction evidence is therefore split into four gzip shards per model/task, each containing five whole partitions. Every original row and decimal score string is retained; provenance is in `provenance/prediction_shards.json`. No fit is split between shards.

Anonymous GitHub changes identity-bearing text, so the original full-repository checksums cannot all remain valid in its download. On the original GitHub repository use:

```bash
python code/verify_repository.py
```

On a downloaded anonymous mirror use:

```bash
python code/verify_repository.py --anonymous-view
```

The second mode skips checksums only for an explicit set of identity-bearing presentation files. Scientific Python files, executable notebook cells, original frozen signatures, data, predictions and metrics still require their original hashes. It is not permission to change experimental code or results. Table/matching replay commands are the same in either view.

Official service documentation: [homepage](https://anonymous.4open.science/) and [FAQ](https://anonymous.4open.science/faq). Creation or anonymization is not an upstream license grant.
