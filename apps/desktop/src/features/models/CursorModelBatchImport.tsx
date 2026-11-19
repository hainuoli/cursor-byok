import { Button } from "../../shared/ui/Button";
import { Checkbox } from "../../shared/ui/Checkbox";
import styles from "./CursorSettings.module.scss";

export type CursorBatchModelOption = {
  modelId: string;
  displayName: string;
  contextWindowTokens: number | null;
  maxOutputTokens: number | null;
  existing: boolean;
};

/** 发现结果的多选区:勾选后把当前供应商的全部模型一次建好,共用表单里的接入配置。 */
export function CursorModelBatchImport({ options, selected, busy, onSelectedChange, onConfirm, onDismiss }: {
  options: CursorBatchModelOption[];
  selected: Set<string>;
  busy: boolean;
  onSelectedChange: (selected: Set<string>) => void;
  onConfirm: () => void;
  onDismiss: () => void;
}) {
  const available = options.filter((option) => !option.existing);
  const allSelected = available.length > 0 && available.every((option) => selected.has(option.modelId));
  const toggle = (modelId: string, checked: boolean) => {
    const next = new Set(selected);
    if (checked) next.add(modelId);
    else next.delete(modelId);
    onSelectedChange(next);
  };
  const toggleAll = (checked: boolean) => onSelectedChange(checked
    ? new Set(available.map((option) => option.modelId))
    : new Set());

  return <div className={styles.batchPanel}>
    <div className={styles.batchHeader}>
      <strong>{t("发现 {count} 个模型", { count: options.length })}</strong>
      <span>{t("勾选的模型会共用上面的服务器地址、API Key 与分组设置。")}</span>
    </div>
    <div className={styles.batchToolbar}>
      <Checkbox checked={allSelected} disabled={available.length === 0} label={t("全选")} onChange={toggleAll} />
      <span>{t("已选 {selected} / {total}", { selected: selected.size, total: available.length })}</span>
    </div>
    <div className={styles.batchList}>
      {options.map((option) => <div key={option.modelId} className={styles.batchRow}>
        <Checkbox
          checked={selected.has(option.modelId)}
          disabled={option.existing}
          label={option.displayName}
          onChange={(checked) => toggle(option.modelId, checked)}
        />
        {option.displayName !== option.modelId && <span className={styles.batchId} title={option.modelId}>{option.modelId}</span>}
        {option.existing && <span className={styles.batchExisting}>{t("已添加")}</span>}
      </div>)}
    </div>
    <div className={styles.batchActions}>
      <Button disabled={busy || selected.size === 0} onClick={onConfirm}>
        {busy ? t("添加中…") : t("添加选中的 {count} 个模型", { count: selected.size })}
      </Button>
      <Button disabled={busy} onClick={onDismiss}>{t("收起")}</Button>
    </div>
  </div>;
}