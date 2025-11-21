import '@vaadin/checkbox';
import '@vaadin/confirm-dialog'
import '@vaadin/checkbox-group';
import '@vaadin/text-field';
import { Notification } from '@vaadin/notification';
import { customElement, property, observe } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import request from '../../../utils/request';
import { isEmpty } from 'lodash';
import i18next from '../../../common/i18n'

@customElement('tf-filter-precision-error')
class TfFilterPrecisionError extends PolymerElement {
    static readonly template = html`
        <vaadin-confirm-dialog
          id="filter-dialog"
          header="[[t('screening_accuracy_error_metric')]]"
          cancel-button-visible
          cancel-text="[[t('cancel')]]"
          confirm-text="[[t('confirm')]]"
          confirm-theme=" primary"
          opened="{{filterDialogOpened}}"
        >
            <vaadin-checkbox-group
                label=""
                value="{{filterValue}}"
                theme="vertical"
            >
                <vaadin-checkbox value="[[MAX_RELATIVE_ERR]]" label="MaxRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[MIN_RELATIVE_ERR]]" label="MinRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[MEAN_RELATIVE_ERR]]" label="MeanRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[NORM_RELATIVE_ERR]]" label="NormRelativeErr"></vaadin-checkbox>
            </vaadin-checkbox-group>
        </vaadin-confirm-dialog>
    `
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);

    @property({ type: Boolean, notify: true })
    filterDialogOpened: boolean = false;

    @property({ type: Array })
    filterValue: string[] = [];

    @property({ type: Object })
    selection: any;

    @property({ type: Object })
    updateFilterData: Function = () => { };

    @property({ type: Array })
    lastFilterValue: string[] = [];

    MAX_RELATIVE_ERR = "0";
    MIN_RELATIVE_ERR = "1";
    MEAN_RELATIVE_ERR = "2";
    NORM_RELATIVE_ERR = "3";

    @observe('selection')
    _selectionChanged() {
        this.set('filterValue', [this.MAX_RELATIVE_ERR, this.MIN_RELATIVE_ERR, this.MEAN_RELATIVE_ERR, this.NORM_RELATIVE_ERR]);
    }
    @observe('filterDialogOpened')
    _onFlterDialogOpened = () => {
        if (this.filterDialogOpened) {
            this.set('lastFilterValue', this.filterValue);
        }

    }

    override ready(): void {
        super.ready();
        const filterDialog = this.shadowRoot?.querySelector('#filter-dialog') as HTMLElement;
        filterDialog?.addEventListener('confirm', this.onFlterDialogConfirm)
        filterDialog?.addEventListener('cancel', this.onFlterDialogCancel)
        this.set('filterValue', [this.MAX_RELATIVE_ERR, this.MIN_RELATIVE_ERR, this.MEAN_RELATIVE_ERR, this.NORM_RELATIVE_ERR]);
    }


    onFlterDialogCancel = (e: any) => {
        this.set('filterValue', this.lastFilterValue);
    }

    onFlterDialogConfirm = async (e: any) => {
        if (isEmpty(this.filterValue)) {
            Notification.show(this.t('precision_error'), {
                position: 'middle',
                duration: 1800,
                theme: 'error',
            });
            this.set('filterValue', this.lastFilterValue);
            setTimeout(() => {
                this.set('filterDialogOpened', true);
            }, 1000)

            return;
        }
        const data = {
            metaData: this.selection,
            filterValue: this.filterValue
        };
        const { success, error } = await request({ url: 'updatePrecisionError', method: 'POST', data });
        if (success) {
            const updateHierarchyData = new CustomEvent('updateHierarchyData', { bubbles: true, composed: true });
            this.dispatchEvent(updateHierarchyData);
            this.set('filterDialogOpened', false);
            this.updateFilterData();
            Notification.show(this.t('percision_update_success'), {
                position: 'middle',
                duration: 2000,
                theme: 'success',
            });
        }
        else {
            Notification.show(`${this.t('caluculate_precision_error')}${error}`, {
                position: 'middle',
                duration: 1800,
                theme: 'error',
            });
            this.set('filterValue', this.lastFilterValue);
            setTimeout(() => {
                this.set('filterDialogOpened', true);
            }, 1000)
        }
    }

}